"""
=============================================================
QUERY NORMALIZER AGENT
=============================================================

Purpose
-------
Transforms noisy user queries into high-quality search queries
for downstream retrieval and reasoning.

Responsibilities
----------------
✓ Fix spelling mistakes
✓ Fix OCR errors
✓ Fix grammar
✓ Preserve unknown entities
✓ Detect intent
✓ Detect comparison
✓ Extract entities
✓ Decide whether retrieval is needed
✓ Return strict validated JSON

Architecture
------------
User Query
      │
      ▼
Preprocessing
      │
      ▼
Fast Rule Engine
      │
      ├────────────► Small-talk
      │
      ├────────────► Fast Intent
      │
      ▼
LLM Rewrite
      │
      ▼
Validation
      │
      ▼
Post Processing
      │
      ▼
Normalized Query

Only ONE LLM call.

Designed for production RAG systems.
"""


from __future__ import annotations

from math import e
import re

from typing import Any
from typing import Dict

from backend.agents_claude.graph_state import GraphState
from backend.agents_claude.llm_client import get_client

VALID_INTENTS = {

    "coverage",

    "exclusion",

    "claim",

    "claim_status",

    "cashless",

    "hospital",

    "network_hospital",

    "waiting_period",

    "renewal",

    "portability",

    "co_payment",

    "deductible",

    "room_rent",

    "premium",

    "sum_insured",

    "grace_period",

    "pre_existing",

    "maternity",

    "comparison",

    "recommendation",

    "definition",

    "general",

}
SMALL_TALK = {

    "hi",

    "hello",

    "hey",

    "thanks",

    "thank you",

    "good morning",

    "good evening",

    "good afternoon",

    "who are you",

    "help",

}

COMPARISON_REGEX = re.compile(

    r"""
    \b(
        vs
        |versus
        |compare
        |comparison
        |difference
        |better
        |which
        |or
    )\b
    """,

    re.I | re.X,

)
INTENT_PATTERNS = {

    "coverage":
    r"\b(cover|covered|coverage|include|included|benefit|eligible)\b",

    "exclusion":
    r"\b(exclusion|excluded|not covered|not cover|denied)\b",

    "claim":
    r"\b(claim|claiming|reimbursement|reimburse)\b",

    "claim_status":
    r"\b(claim status|track claim|claim progress|status)\b",

    "cashless":
    r"\b(cashless)\b",

    "hospital":
    r"\b(hospital)\b",

    "network_hospital":
    r"\b(network hospital|empanelled hospital)\b",

    "waiting_period":
    r"\b(waiting period|waiting|cooling period)\b",

    "renewal":
    r"\b(renew|renewal)\b",

    "portability":
    r"\b(portability|port policy|switch policy)\b",

    "co_payment":
    r"\b(co-payment|co payment|copay)\b",

    "deductible":
    r"\b(deductible)\b",

    "room_rent":
    r"\b(room rent|icu rent)\b",

    "premium":
    r"\b(premium|price|cost|amount)\b",

    "sum_insured":
    r"\b(sum insured|coverage amount|insured amount)\b",

    "grace_period":
    r"\b(grace period)\b",

    "pre_existing":
    r"\b(pre existing|pre-existing)\b",

    "maternity":
    r"\b(maternity|pregnancy|delivery)\b",

    "comparison":
    r"\b(compare|comparison|difference|vs|versus)\b",

    "definition":
    r"\b(what is|meaning|define|definition)\b",

}

SYSTEM_PROMPT = """
You are an Expert Insurance Query Understanding Engine.

## ROLE

You are the first stage of an Insurance RAG pipeline.

Your output is NOT shown to the user.

It is used ONLY for semantic retrieval over insurance policy documents.

Therefore your job is to rewrite the user's query into the BEST possible search query while preserving every important entity.

DO NOT answer the user's question.

---

## YOUR RESPONSIBILITIES

1. Correct spelling mistakes.

2. Correct OCR mistakes.

3. Correct broken words.

4. Correct grammar.

5. Expand obvious abbreviations.

6. Rewrite into ONE complete standalone English question.

7. Preserve every important entity.

8. Detect user intent.

9. Detect comparison queries.

10. Decide whether retrieval is required.

11. Extract structured entities.

12. Return ONLY valid JSON.

---

## ENTITY PRESERVATION (CRITICAL)

Never change, replace or invent the following:

• insurance company
• insurer
• organization
• policy name
• plan name
• product name
• hospital
• doctor
• disease
• medical condition
• surgery
• treatment
• medicine
• medical procedure
• numbers
• percentages
• ages
• money values
• dates
• waiting periods

If you DO NOT recognize an entity,

KEEP IT EXACTLY AS WRITTEN.

Never replace an unknown company with a known company.

Never replace an unknown disease with another disease.

Never guess.

---

## QUERY REWRITING RULES

GOOD

"waht is waitng perod for catarct"

↓

"What is the waiting period for cataract surgery?"

BAD

"Waiting period?"

↓

(do not make incomplete queries)

---

## INTENT CLASSES

Return exactly ONE of:

coverage
exclusion
claim
claim_status
cashless
hospital
network_hospital
waiting_period
renewal
portability
co_payment
deductible
room_rent
premium
sum_insured
grace_period
pre_existing
maternity
comparison
recommendation
definition
general

---

## INTENT GUIDELINES

coverage
Questions asking whether something is covered.

claim
Questions about filing, reimbursement or claim process.

claim_status
Questions asking claim progress or status.

waiting_period
Questions asking about waiting periods.

cashless
Questions about cashless treatment.

hospital
Questions about hospitals.

network_hospital
Questions asking whether a hospital belongs to the insurer's network.

premium
Questions asking premium or price.

renewal
Questions about policy renewal.

comparison
Comparing two or more insurers, policies or plans.

definition
Questions asking meaning of an insurance term.

general
Everything else.

---

## COMPARISON DETECTION

comparison=true when the user is

comparing

choosing

asking "better"

asking "difference"

asking "vs"

asking "versus"

asking "which"

asking "or"

Examples

Star vs HDFC

Care or Niva

Which is better

Difference between ABC and XYZ

---

## RETRIEVAL

needs_retrieval=false ONLY when the query is

hello

hi

hey

good morning

thanks

thank you

who are you

help

Otherwise

needs_retrieval=true

---

## OUTPUT FORMAT

Return ONLY valid JSON.

Do not return Markdown.

Do not explain anything.

Return exactly this schema.

{
"normalized_query": "",
"intent": "",
"intent_confidence": 0.95,
"comparison": false,
"needs_retrieval": true,
"entities": {
"organizations": [],
"policies": [],
"conditions": [],
"procedures": [],
"medicines": [],
"hospitals": [],
"persons": [],
"numbers": []
}
}

---

## EXAMPLES

INPUT
waht is waitng perod for catarct

OUTPUT
{
"normalized_query":"What is the waiting period for cataract surgery?",
"intent":"waiting_period",
"intent_confidence":0.99,
"comparison":false,
"needs_retrieval":true,
"entities":{
"organizations":[],
"policies":[],
"conditions":["cataract"],
"procedures":["cataract surgery"],
"medicines":[],
"hospitals":[],
"persons":[],
"numbers":[]
}
}

INPUT
does hdfc ergo c0ver diabtes

OUTPUT
{
"normalized_query":"Does HDFC ERGO cover diabetes?",
"intent":"coverage",
"intent_confidence":0.99,
"comparison":false,
"needs_retrieval":true,
"entities":{
"organizations":["HDFC ERGO"],
"policies":[],
"conditions":["diabetes"],
"procedures":[],
"medicines":[],
"hospitals":[],
"persons":[],
"numbers":[]
}
}

INPUT
Star vs Niva Bupa

OUTPUT
{
"normalized_query":"Compare Star Health and Niva Bupa health insurance.",
"intent":"comparison",
"intent_confidence":0.99,
"comparison":true,
"needs_retrieval":true,
"entities":{
"organizations":["Star Health","Niva Bupa"],
"policies":[],
"conditions":[],
"procedures":[],
"medicines":[],
"hospitals":[],
"persons":[],
"numbers":[]
}
}

INPUT
What is deductible

OUTPUT
{
"normalized_query":"What is a deductible in health insurance?",
"intent":"definition",
"intent_confidence":0.98,
"comparison":false,
"needs_retrieval":true,
"entities":{
"organizations":[],
"policies":[],
"conditions":[],
"procedures":[],
"medicines":[],
"hospitals":[],
"persons":[],
"numbers":[]
}
}

INPUT
Does XYZ Elite Plus cover ABC syndrome

OUTPUT
{
"normalized_query":"Does XYZ Elite Plus cover ABC syndrome?",
"intent":"coverage",
"intent_confidence":0.87,
"comparison":false,
"needs_retrieval":true,
"entities":{
"organizations":["XYZ"],
"policies":["Elite Plus"],
"conditions":["ABC syndrome"],
"procedures":[],
"medicines":[],
"hospitals":[],
"persons":[],
"numbers":[]
}
}

INPUT
hello

OUTPUT
{
"normalized_query":"hello",
"intent":"general",
"intent_confidence":1.0,
"comparison":false,
"needs_retrieval":false,
"entities":{
"organizations":[],
"policies":[],
"conditions":[],
"procedures":[],
"medicines":[],
"hospitals":[],
"persons":[],
"numbers":[]
}
}

Return ONLY valid JSON.
"""

def fast_intent(query: str) -> str:

    q = query.lower()

    for intent, pattern in INTENT_PATTERNS.items():

        if re.search(pattern, q):

            return intent

    return "general"

def preprocess(query: str) -> str:

    query = query.strip()

    query = re.sub(r"\s+", " ", query)

    query = query.replace("|", " ")

    query = query.replace("•", " ")

    query = query.replace("—", "-")

    query = re.sub(r"\?{2,}", "?", query)

    query = re.sub(r"\.{2,}", ".", query)

    query = re.sub(r",{2,}", ",", query)

    ocr_fixes = {

    "c0ver": "cover",

    "per0d": "period",

    "h0spital": "hospital",

    "p0licy": "policy",

    "clalm": "claim",

    "lnsurance": "insurance",

    "beneflt": "benefit",

    "dlabetes": "diabetes",

}

    for wrong, correct in ocr_fixes.items():
        query = re.sub(
        wrong,
        correct,
        query,
        flags=re.I,
    )

    return query

def validate_output(result: Dict[str, Any], raw: str):

    defaults = {

        "normalized_query": raw,

        "intent": "general",

        "intent_confidence": 0.50,

        "comparison": False,

        "needs_retrieval": True,

        "entities": {

            "organizations": [],

            "policies": [],

            "conditions": [],

            "procedures": [],

            "medicines": [],

            "hospitals": [],

            "persons": [],

            "numbers": [],

        },

    }

    if not isinstance(result, dict):

        return defaults

    for key in defaults:

        if key in result:

            defaults[key] = result[key]

    if defaults["intent"] not in VALID_INTENTS:

        defaults["intent"] = "general"

    defaults["comparison"] = bool(defaults["comparison"])

    defaults["needs_retrieval"] = bool(defaults["needs_retrieval"])

    if not isinstance(defaults["entities"], dict):

        defaults["entities"] = {

            "organizations": [],

            "policies": [],

            "conditions": [],

            "procedures": [],

            "medicines": [],

            "hospitals": [],

            "persons": [],

            "numbers": [],

        }

    return defaults

# =============================================================
# QUERY NORMALIZER AGENT
# =============================================================

class QueryNormalizerAgent:

    def __init__(self):

        self.llm = get_client()

    def run(self, state: GraphState) -> Dict[str, Any]:

        raw = state.get("raw_query", "").strip()

        if not raw:

            return {

                "normalized_query": "",

                "intent": "general",

                "intent_confidence": 0.0,

                "comparison": False,

                "needs_retrieval": False,

                "entities": {

                    "organizations": [],

                    "policies": [],

                    "conditions": [],

                    "procedures": [],

                    "medicines": [],

                    "hospitals": [],

                    "persons": [],

                    "numbers": [],

                }

            }

        ###############################################################
        # PREPROCESS
        ###############################################################

        raw = preprocess(raw)

        lower = raw.lower()

        ###############################################################
        # SMALL TALK
        ###############################################################
        if any(lower.startswith(x) for x in SMALL_TALK):
        

            return {

                "normalized_query": raw,

                "intent": "general",

                "intent_confidence": 1.0,

                "comparison": False,

                "needs_retrieval": False,

                "entities": {

                    "organizations": [],

                    "policies": [],

                    "conditions": [],

                    "procedures": [],

                    "medicines": [],

                    "hospitals": [],

                    "persons": [],

                    "numbers": [],

                }

            }

        ###############################################################
        # FAST RULE ENGINE
        ###############################################################

        
        rule_intent = fast_intent(raw)

        is_comparison = bool(COMPARISON_REGEX.search(lower))

        ###############################################################
        # CLEAN QUERY?
        ###############################################################

        looks_clean = (

            raw.endswith("?")

            and raw[0].isupper()

            and len(raw.split()) >= 6

            and rule_intent != "general"

        )

        ###############################################################
        # FAST PATH
        ###############################################################

        if looks_clean:

            return {

                "normalized_query": raw,

                "intent": "comparison" if is_comparison else rule_intent,

                "intent_confidence": 0.98,

                "comparison": is_comparison,

                "needs_retrieval": True,

                "entities": {

                    "organizations": [],

                    "policies": [],

                    "conditions": [],

                    "procedures": [],

                    "medicines": [],

                    "hospitals": [],

                    "persons": [],

                    "numbers": [],

                }

            }

        ###############################################################
        # LLM CALL
        ###############################################################

        messages = [

            {

                "role": "system",

                "content": SYSTEM_PROMPT,

            },

            {

                "role": "user",

                "content": raw,

            }

        ]
        result = {}

        try:

            result = self.llm.call_json(

                messages,

                max_tokens=350,

            )

        except Exception:
            print("\nLLM ERROR:")
            print(e)

            result = {}
            
        print("\nRaw LLM Output")
        print(result)

        ###############################################################
        # VALIDATION
        ###############################################################

        result = validate_output(

            result,

            raw,

        )

        ###############################################################
        # POST PROCESSING
        ###############################################################

        q = result["normalized_query"].strip()
        
        if q:
            q = q.strip()

            q = q[0].upper() + q[1:]

            q = q.rstrip(".!?")

            q += "?"

        

        result["normalized_query"] = q

        result["comparison"] = bool(

            result.get(

                "comparison",

                False,

            )

        )

        result["needs_retrieval"] = bool(

            result.get(

                "needs_retrieval",

                True,

            )

        )

        ###############################################################
        # HYBRID INTENT CHECK
        ###############################################################

        
        regex_intent = rule_intent

        llm_intent = result["intent"]

        confidence = float(

            result.get(

                "intent_confidence",

                0.5,

            )

        )

        if (

            regex_intent != "general"

            and confidence < 0.75

        ):

            result["intent"] = regex_intent

            result["intent_confidence"] = 0.75

        ###############################################################
        # COMPARISON ENFORCEMENT
        ###############################################################

        if is_comparison:

            result["comparison"] = True

            result["intent"] = "comparison"

        ###############################################################
        # FINAL SAFETY
        ###############################################################

        if result["intent"] not in VALID_INTENTS:

            result["intent"] = "general"

        return result


# =============================================================
# LANGGRAPH NODE
# =============================================================

_AGENT = None


def query_normalizer_node(

    state: GraphState,

) -> GraphState:

    global _AGENT

    if _AGENT is None:

        _AGENT = QueryNormalizerAgent()

    updates = _AGENT.run(state)

    return {

        **state,

        **updates,

    }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    import json

    agent = QueryNormalizerAgent()

    TEST_QUERIES = [

        "waht is waitng perod for catarct",

        "does hdfc ergo c0ver diabtes",

        "star vs hdfc health insurance",

        "what is not coverd in this policy",

        "matrnity benifit in niva bupa",

        "claim after 30 day possible",

        "hello",

        "thank you",

        "what is deductible",

        "Apollo Munich cover cancer",

        "XYZ Insurance Elite plan catarct surgery",

        "cashless hospital near me",

        "renew policy",

        "compare care vs hdfc",

        "pre existing diabetes waiting period",

    ]

    print("\n" + "=" * 70)

    print("QUERY NORMALIZER TEST")

    print("=" * 70)

    for q in TEST_QUERIES:

        result = agent.run(

            {

                "raw_query": q,

            }

        )

        print("\nINPUT : ", q)

        print(

            json.dumps(

                result,

                indent=4,

            )

        )

        print("-" * 70)

    print()

    print(

        "Cache Stats:",

        agent.llm.cache_stats(),

    )