# Inside the Agent Loop: Hands-On with CrewAI Before Agent Studio


This lab walks you through building a 2-agent CrewAI crew inside a **Cloudera CML JupyterLab session** using **Azure OpenAI** as the LLM backend.

Because Cloudera Agent Studio is built on top of CrewAI internally, completing this exercise first means you understand exactly what Agent Studio automates for you.

---

## What you will build

A **News Research Crew** with two agents:

| Agent | Role | Tool | Output |
|---|---|---|---|
| Researcher | Senior Research Analyst | Web Search (Mock) | Numbered findings |
| Writer | Content Strategist | None | 3-paragraph brief |

---

## Requirements

- Cloudera CML JupyterLab session (Python 3.10)
- Azure OpenAI deployment (gpt-4o)
- CrewAI 1.14.1

---

## Step 1 — Set CML Environment Variables

In **CML → Project Settings → Advanced → Environment Variables**, add:

| Variable | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key (**mark as Secret**) |
| `AZURE_OPENAI_ENDPOINT` | e.g. `https://<your-resource>.openai.azure.com` |
| `AZURE_OPENAI_API_VERSION` | e.g. `2024-08-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | e.g. `gpt-4o` |

> ⚠️ After saving, **stop and restart your CML session** — environment variables are only injected at session start.

---

## Step 2 — Open the notebook

Upload `Lab_A_Inside_The_Agent_Loop.ipynb` into your CML JupyterLab session and run each cell in order.

Or copy the cells below manually.

---

## Notebook cells

### Cell 0 — Install CrewAI

> ⚠️ After this cell finishes, click **Kernel → Restart Kernel** before running any other cell.

```python
%pip install crewai==1.14.1 --quiet
```

---

### Cell 1 — Verify CrewAI installation

```python
import crewai
print(f"CrewAI version: {crewai.__version__}")  # should print 1.14.1
```

---

### Cell 2 — Verify Azure environment variables

```python
import os

required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
]

print("Environment variable check:")
all_ok = True
for var in required_vars:
    val = os.environ.get(var, "")
    status = "OK" if val else "MISSING"
    if not val:
        all_ok = False
    display = "*" * 8 + val[-4:] if "KEY" in var else val
    print(f"  {status:8s}  {var}: {display}")

print()
if all_ok:
    print("✅ All variables set. Ready to proceed.")
else:
    print("❌ Some variables are missing. Fix them before continuing.")
```

---

### Cell 3 — Configure AzureDirectLLM

> **Why not use `LLM(model='azure/gpt-4o')` directly?**
> CrewAI 1.14.1 has a bug in its native Azure provider — it requires `azure-ai-inference`
> which is not available in this CML runtime. `AzureDirectLLM` calls Azure directly
> via `requests`, bypassing the broken routing entirely.

```python
import os
import requests
from crewai.llms.base_llm import BaseLLM

# Read all credentials from CML environment variables — no hardcoded values
AZURE_ENDPOINT    = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
AZURE_DEPLOYMENT  = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_API_KEY     = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_URL         = (
    f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}"
    f"/chat/completions?api-version={AZURE_API_VERSION}"
)


class AzureDirectLLM(BaseLLM):
    """Calls Azure OpenAI directly via requests.
    Bypasses CrewAI 1.14.1 native Azure provider bug.
    """
    model: str = "azure/gpt-4o"  # required field in BaseLLM (Pydantic v2)

    def call(self, messages, tools=None, **kwargs):
        payload = {
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.3,
        }
        response = requests.post(
            AZURE_URL,
            headers={"Content-Type": "application/json", "api-key": AZURE_API_KEY},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        if message.get("tool_calls"):
            return message["tool_calls"]
        return message["content"]

    def supports_function_calling(self):
        return False  # forces ReAct text loop — simpler and more observable


# Instantiate — pass model= explicitly (Pydantic v2 requirement in CrewAI 1.14.1)
llm = AzureDirectLLM(model="azure/gpt-4o")
print("✅ LLM ready:", llm.model)
```

---

### Cell 4 — Define MockSearchTool

> `DuckDuckGoSearchRun` has import conflicts on this CML runtime.
> The mock tool produces the same learning outcome — watching the agent
> decide to call a tool and process the result.

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(description="Search query string")


class MockSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Searches for information on a given topic"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return (
            f"Search results for '{query}':\n"
            "1. Agentic AI is transforming enterprise data workflows in 2025.\n"
            "2. Multi-agent frameworks like CrewAI are widely adopted in production.\n"
            "3. Cloudera Agent Studio provides enterprise-grade agent orchestration.\n"
            "4. RAG remains the dominant pattern for grounding agent responses.\n"
            "5. MCP protocol is emerging as the standard for tool integration."
        )


search_tool = MockSearchTool()
print("✅ Tools ready:", search_tool.name)
```

---

### Cell 5 — Define Researcher and Writer agents

```python
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find accurate, current information on {topic}",
    backstory=(
        "You are an expert analyst who uncovers clear, factual insights. "
        "You always cite key points and avoid speculation."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,  # prevents infinite loops in CML sessions
)

writer = Agent(
    role="Content Strategist",
    goal="Turn research into a concise 3-paragraph professional brief",
    backstory=(
        "You are a skilled writer who transforms raw research into clear, "
        "professional briefings. You never invent facts."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

print("✅ Agents defined:", researcher.role, "|", writer.role)
```

---

### Cell 6 — Define Tasks

```python
research_task = Task(
    description=(
        "Search for recent developments on {topic}. "
        "Identify the 5 most important facts or trends. "
        "Present findings as a numbered list, one sentence per point."
    ),
    expected_output="A numbered list of 5 key findings about {topic}",
    agent=researcher,
)

write_task = Task(
    description=(
        "Using the researcher's findings, write a 3-paragraph professional "
        "briefing on {topic}. "
        "Para 1: overview. Para 2: key developments. Para 3: implications."
    ),
    expected_output="A polished 3-paragraph briefing on {topic}",
    agent=writer,
    context=[research_task],  # passes researcher output to writer
)

print("✅ Tasks defined.")
```

---

### Cell 7 — Assemble Crew and Run

> Change the `topic` value to anything relevant to your project.

```python
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

try:
    result = crew.kickoff(
        inputs={"topic": "Agentic AI in enterprise data platforms 2025"}
    )
    print("\n" + "=" * 70)
    print(" FINAL BRIEFING")
    print("=" * 70)
    print(str(result))
except Exception as e:
    print(f"Error type : {type(e).__name__}")
    print(f"Error      : {e}")
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `KeyError: AZURE_OPENAI_API_KEY` | Env var not set. Add it in CML Project Settings and restart the session. |
| `ValidationError: Model name is required` | Pass model explicitly: `AzureDirectLLM(model="azure/gpt-4o")`. Already in Cell 3. |
| `401 PermissionDenied` | API key is incorrect. Verify the value in CML Project Settings. |
| `404 Resource not found` | Endpoint URL or deployment name is wrong. Endpoint must not include a trailing path. |
| `ImportError: cannot import OutputParserError` | Two conflicting crewai versions installed. Run `pip uninstall crewai -y`, restart session, reinstall. |
| `ModuleNotFoundError: No module named crewai` | Run `%pip install crewai==1.14.1 --quiet` then restart the kernel. |
| `Azure AI Inference native provider not available` | Do not use `LLM(model='azure/gpt-4o')`. Use `AzureDirectLLM` from Cell 3 instead. |

---

## Bridge to Agent Studio

Every concept in this lab maps directly to Cloudera Agent Studio:

| What you wrote in CrewAI | Where it lives in Agent Studio |
|---|---|
| `Agent(role, goal, backstory)` | Agent config panel |
| `tools=[search_tool]` | Tool library → assign to agent |
| `Task(description, expected_output)` | Step / Action editor |
| `context=[research_task]` | Task dependency arrow on canvas |
| `Crew(agents, tasks, process)` | Pipeline canvas |
| `crew.kickoff(inputs={...})` | Run / Test button |
| `verbose=True` output | Trace viewer |
| `os.environ["AZURE_..."]` | Same CML project env vars — reused automatically |

> **Key take-away:** The Azure environment variables you set in CML for this lab are the exact same variables Agent Studio uses. There is nothing to reconfigure when you move from notebook to platform.

---

## Bonus challenges

**Challenge A — Change the topic**

```python
result = crew.kickoff(inputs={"topic": "Cloudera Data Platform on Azure cost optimisation"})
```

**Challenge B — Add a third Reviewer agent**

Add a `Fact Checker` agent after the Writer that verifies the brief contains no invented claims. Add it to the Crew's agents list and create a Task with `context=[write_task]`.

**Challenge C — Switch to hierarchical process**

```python
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.hierarchical,
    manager_llm=llm,
    verbose=True,
)
```

Watch how the delegation pattern changes in the verbose output compared to sequential.

---

*Cloudera CDAI Practice · Agentic AI Training · Lab A · CML + Azure OpenAI · CrewAI 1.14.1*
