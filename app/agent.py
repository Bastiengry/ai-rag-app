"""
Agent LangGraph pour l'assistant RAG local.

Graphe :
    START → router ┬─ documents → search → generate → verify ─┬─ finish → END
                   │                 ▲                        │
                   │                 └──── retry (1 fois) ────┘
                   ├─ calculator → END
                   ├─ date → END
                   └─ direct → END

Patterns couverts : routage / choix d'outil, Planner→Executor simplifié,
boucle de réflexion (auto-évaluation + nouvelle recherche), sorties JSON validées (Pydantic).
"""
import ast
import operator
import os
import re
from datetime import datetime
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

MAX_RETRIES = 1  # nombre de nouvelles recherches autorisées après une vérification négative


# --------------------------------------------------
# Sorties structurées (validées par Pydantic)
# --------------------------------------------------
class RouteDecision(BaseModel):
    route: Literal["documents", "calculator", "date", "direct"]
    tool_input: str = ""


class Verdict(BaseModel):
    grounded: bool
    reason: str = ""
    better_query: str = ""


def ask_json(llm_json, model_cls, system, user, default, retries=2):
    """Interroge le LLM en mode JSON, valide la sortie, réessaie, puis retombe sur `default`."""
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    for _ in range(retries):
        raw = llm_json.invoke(messages).content
        try:
            return model_cls.model_validate_json(raw)
        except ValueError:  # inclut pydantic.ValidationError
            continue
    return default


# --------------------------------------------------
# Outils
# --------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exposant trop grand")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Expression non autorisée")


def calculator(expression: str) -> float:
    """Évalue une expression arithmétique de façon sûre (pas de eval)."""
    expression = expression.replace(",", ".").replace("×", "*").replace("^", "**")
    return _eval(ast.parse(expression.strip(), mode="eval").body)


JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]


def current_date() -> str:
    now = datetime.now()
    return f"{JOURS[now.weekday()]} {now.day} {MOIS[now.month - 1]} {now.year}, {now:%H:%M}"


# --------------------------------------------------
# Détection déterministe (chemin rapide, sans LLM)
# --------------------------------------------------
_MATH_RUN = re.compile(r"[\d(][\d\s.,+\-*/^%()]*[\d)]")
_DATE_LIKE = re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b")
_CALC_CUES = ("calcul", "combien", "résultat", "resultat", "=", "font", "fait")
_DATE_QUESTION = re.compile(
    r"(quel jour (sommes|est|on)|quelle heure|heure est.il|date d.aujourd"
    r"|on est le combien|nous sommes le combien|sommes-nous le)"
)


def extract_expression(text):
    """Retourne une expression arithmétique valide trouvée dans `text`, sinon None."""
    t = text.replace("×", "*").replace("÷", "/")
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", t, flags=re.I)
    if _DATE_LIKE.search(t):  # évite de prendre 12/05/2024 pour une division
        return None
    best = None
    for m in _MATH_RUN.finditer(t):
        cand = m.group().strip()
        if not re.search(r"\d\s*[-+*/^%]\s*[\d(]", cand):  # il faut un opérateur entre deux opérandes
            continue
        try:
            calculator(cand)
        except Exception:
            continue
        if best is None or len(cand) > len(best):
            best = cand
    if not best:
        return None
    low = t.lower()
    if any(c in low for c in _CALC_CUES) or len(best) >= 0.4 * len(t.strip()):
        return best
    return None


# --------------------------------------------------
# Prompts
# --------------------------------------------------
ROUTER_PROMPT = """Tu classes la question d'un utilisateur pour choisir un outil.

Outils :
- "documents" : la question porte sur le contenu de documents (PDF, Word) que l'utilisateur a indexés. C'est le cas par défaut.
- "calculator" : la question est un calcul numérique. tool_input = l'expression mathématique seule, avec des chiffres et + - * / ( ).
- "date" : la question demande la date ou l'heure actuelle.
- "direct" : salutation, remerciement ou question de culture générale sans lien avec des documents.

Exemples :
Question : Quelles sont les destinations au départ de Bâle ?
{"route": "documents", "tool_input": ""}
Question : Combien font 245 multiplié par 18 ?
{"route": "calculator", "tool_input": "245*18"}
Question : Quel jour sommes-nous ?
{"route": "date", "tool_input": ""}
Question : Bonjour, merci pour ton aide !
{"route": "direct", "tool_input": ""}

Réponds uniquement avec un objet JSON, sans autre texte."""

REWRITE_PROMPT = (
    "Reformule la dernière question de l'utilisateur en une question autonome, compréhensible "
    "sans l'historique (remplace les pronoms et références par ce qu'ils désignent). "
    "Si elle est déjà autonome, recopie-la telle quelle. "
    "Réponds uniquement par la question, sans commentaire."
)

ANSWER_PROMPT = (
    "Tu es un assistant expert.\n"
    "Réponds UNIQUEMENT en utilisant le contexte fourni ci-dessous.\n"
    "Sois direct et précis. Si le document affirme quelque chose, même brièvement, "
    "utilise cette information.\n"
    "Si la réponse ne se trouve pas dans le contexte, dis clairement que tu ne trouves "
    "pas l'information dans les documents. N'invente rien.\n"
)

VERIFIER_PROMPT = """Tu es un vérificateur strict. On te donne une QUESTION, un CONTEXTE et une RÉPONSE.
Vérifie que chaque affirmation de la RÉPONSE est appuyée par le CONTEXTE.
- grounded = true si la réponse est fidèle au contexte, ou si elle dit honnêtement que l'information est absente.
- grounded = false si la réponse contient des éléments absents du contexte ou le contredit.
Si grounded = false, explique le problème dans "reason" et propose dans "better_query" une requête de recherche courte et plus précise.

Réponds uniquement avec un objet JSON, par exemple :
{"grounded": false, "reason": "la durée annoncée n'apparaît pas dans le contexte", "better_query": "durée du contrat"}"""

DIRECT_PROMPT = (
    "Tu es un assistant local sympathique et concis. Réponds en français. "
    "Si la question semble porter sur les documents de l'utilisateur, invite-le à la reformuler précisément."
)


# --------------------------------------------------
# État du graphe
# --------------------------------------------------
class AgentState(TypedDict, total=False):
    question: str
    history: list      # [{"role": "user"|"assistant", "content": str}]
    route: str
    tool_input: str
    context: str
    sources: list
    answer: str
    grounded: bool
    critique: str
    attempts: int
    log: str           # message affiché dans l'interface à chaque étape


def _to_messages(history):
    out = []
    for m in history or []:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        out.append(cls(content=m["content"]))
    return out


def _history_text(history):
    lines = []
    for m in history or []:
        who = "Utilisateur" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content'][:300]}")
    return "\n".join(lines)


# --------------------------------------------------
# Construction de l'agent
# --------------------------------------------------
def build_agent(llm, llm_json, retriever):
    """llm : modèle de chat normal. llm_json : même modèle avec format="json". retriever : ParentDocumentRetriever."""

    def search_documents(query):
        docs = retriever.invoke(query)
        parts, sources = [], []
        for i, doc in enumerate(docs):
            src = os.path.basename(doc.metadata.get("source", "Inconnu"))
            if src not in sources:
                sources.append(src)
            parts.append(f"Document {i + 1} (Source: {src}):\n{doc.page_content}\n")
        return "\n".join(parts), sources

    # ----- Nœuds -----
    def standalone_query(question, history):
        user = f"Historique :\n{_history_text(history)}\n\nDernière question : {question}"
        out = llm.invoke([SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=user)]).content
        out = out.strip().strip('"')
        return out if 0 < len(out) <= 300 else question

    def route_node(state: AgentState):
        question = state["question"]
        history = state.get("history")

        # 1) Chemins rapides déterministes
        expr = extract_expression(question)
        if expr:
            return {"route": "calculator", "tool_input": expr, "attempts": 0,
                    "log": "🧭 Outil choisi : **calculator** (détecté directement)"}
        if _DATE_QUESTION.search(question.lower()):
            return {"route": "date", "tool_input": "", "attempts": 0,
                    "log": "🧭 Outil choisi : **date** (détecté directement)"}

        # 2) Classification par le LLM, sur la question seule (sans historique)
        decision = ask_json(llm_json, RouteDecision, ROUTER_PROMPT,
                            f"Question : {question}", RouteDecision(route="documents"))
        route, tool_input = decision.route, question

        if route == "calculator":
            tool_input = decision.tool_input.strip()
            try:
                calculator(tool_input)
            except Exception:  # expression inutilisable : mieux vaut chercher dans les documents
                route, tool_input = "documents", question

        # 3) Question de suivi courte : on la rend autonome grâce à l'historique
        if route == "documents" and history and len(question.split()) <= 8:
            tool_input = standalone_query(question, history)

        return {"route": route, "tool_input": tool_input, "attempts": 0,
                "log": f"🧭 Outil choisi : **{route}**"}

    def search_node(state: AgentState):
        query = state.get("tool_input") or state["question"]
        context, sources = search_documents(query)
        if not context:
            return {
                "context": "",
                "sources": [],
                "answer": "Je ne trouve aucune information correspondante dans les documents indexés.",
                "log": f"🔍 Recherche « {query} » : aucun passage trouvé",
            }
        return {
            "context": context,
            "sources": sources,
            "log": f"🔍 Recherche « {query} » : {len(sources)} source(s) consultée(s)",
        }

    def generate_node(state: AgentState):
        system = ANSWER_PROMPT
        if state.get("critique"):
            system += (
                f"\nUne vérification précédente a signalé ce problème : {state['critique']}\n"
                "Corrige-le en te limitant strictement au contexte.\n"
            )
        system += f"\nCONTEXTE :\n{state['context']}"
        messages = [SystemMessage(content=system)]
        messages += _to_messages(state.get("history"))
        messages.append(HumanMessage(content=state["question"]))
        answer = llm.invoke(messages).content
        return {"answer": answer, "log": "✍️ Rédaction de la réponse"}

    def verify_node(state: AgentState):
        user = (
            f"QUESTION : {state['question']}\n\n"
            f"CONTEXTE :\n{state['context'][:6000]}\n\n"
            f"RÉPONSE :\n{state['answer']}"
        )
        default = Verdict(grounded=True, reason="vérification indisponible")
        verdict = ask_json(llm_json, Verdict, VERIFIER_PROMPT, user, default)
        attempts = state.get("attempts", 0) + 1
        if verdict.grounded:
            log = "🧪 Vérification : réponse fidèle au contexte"
        else:
            log = f"🧪 Vérification : problème détecté ({verdict.reason or 'non précisé'})"
        return {
            "grounded": verdict.grounded,
            "critique": verdict.reason,
            "tool_input": verdict.better_query.strip() or state.get("tool_input", ""),
            "attempts": attempts,
            "log": log,
        }

    def finish_node(state: AgentState):
        answer = state["answer"]
        if not state.get("grounded", True):
            answer += "\n\n⚠️ *Cette réponse n'a pas pu être entièrement vérifiée à partir des documents.*"
        if state.get("sources"):
            answer += f"\n\n📚 *Sources : {', '.join(state['sources'])}*"
        return {"answer": answer}

    def calc_node(state: AgentState):
        expr = state["tool_input"]
        try:
            result = calculator(expr)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return {"answer": f"{expr} = **{result}**", "log": f"🧮 Calcul : {expr}"}
        except Exception as e:
            return {
                "answer": f"Je n'ai pas pu calculer « {expr} » ({e}).",
                "log": "🧮 Calcul impossible",
            }

    def date_node(state: AgentState):
        return {"answer": f"Nous sommes le {current_date()}.", "log": "📅 Date consultée"}

    def direct_node(state: AgentState):
        messages = [SystemMessage(content=DIRECT_PROMPT)]
        messages += _to_messages(state.get("history"))
        messages.append(HumanMessage(content=state["question"]))
        return {"answer": llm.invoke(messages).content, "log": "💬 Réponse directe"}

    # ----- Décisions -----
    def after_search(state: AgentState):
        return "generate" if state.get("context") else "end"

    def after_verify(state: AgentState):
        if not state.get("grounded", True) and state.get("attempts", 0) <= MAX_RETRIES:
            return "retry"
        return "finish"

    # ----- Assemblage -----
    graph = StateGraph(AgentState)
    graph.add_node("router", route_node)
    graph.add_node("search", search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("finish", finish_node)
    graph.add_node("calc", calc_node)
    graph.add_node("date", date_node)
    graph.add_node("direct", direct_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"documents": "search", "calculator": "calc", "date": "date", "direct": "direct"},
    )
    graph.add_conditional_edges("search", after_search, {"generate": "generate", "end": END})
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", after_verify, {"retry": "search", "finish": "finish"})
    graph.add_edge("finish", END)
    graph.add_edge("calc", END)
    graph.add_edge("date", END)
    graph.add_edge("direct", END)

    return graph.compile()


def stream_agent(graph, question, history):
    """Génère (nom_du_nœud, mise_à_jour) à chaque étape, pour afficher la progression."""
    inputs = {"question": question, "history": history or [], "attempts": 0}
    for update in graph.stream(inputs, stream_mode="updates"):
        for node, delta in update.items():
            yield node, delta
