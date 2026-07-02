# Project Demo Videos

This folder contains the demonstration videos for our **AI Insurance Policy Assistant** project.

## 📹 Demo 1 – Project Walkthrough

This video demonstrates the core functionalities of the system, including:

- Querying the existing insurance policy knowledge base.
- Using **short-term conversational memory**, allowing follow-up questions based on previous queries.
- Uploading a new insurance policy PDF and querying its contents.
- **Duplicate PDF detection**, where the system avoids storing duplicate chunks if the uploaded document already exists in the database and instead answers using the existing indexed data.
- **Human-in-the-Loop (HITL)** workflow, where the LLM asks the user for clarification whenever the query confidence is low before continuing the retrieval and reasoning process.

## 📊 Demo 2 – LangSmith Tracing

This video showcases the **LangSmith** integration used for observability and debugging.

It demonstrates how a single query execution is traced through the complete LangGraph workflow, allowing developers to inspect:

- Node execution order
- Inputs and outputs of each node
- State transitions
- Latency of each step
- Errors (if any)
- Complete execution flow for debugging and performance analysis

## 📁 Video Location

The demonstration videos are available inside the **`assets/`** folder located within this directory.
