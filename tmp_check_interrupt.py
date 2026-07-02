from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    x: str
    y: str

def node(state):
    return {'x': 'done'}

def ask(state):
    value = interrupt('need input')
    return {'y': value}

g = StateGraph(State)
g.add_node('ask', ask)
g.add_node('node', node)
g.set_entry_point('ask')
g.add_edge('ask', 'node')
g.add_edge('node', END)
compiled = g.compile(checkpointer=MemorySaver())

try:
    print(compiled.invoke({'x': '', 'y': ''}, config={'configurable': {'thread_id': 't1'}}))
except Exception as e:
    print(type(e).__name__, e)

try:
    print(compiled.invoke(Command(resume='hello'), config={'configurable': {'thread_id': 't1'}}))
except Exception as e:
    print('resume err', type(e).__name__, e)
