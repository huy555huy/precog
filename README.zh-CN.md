# PreCog

[English](README.md)

面向 Python agent 运行时的推测式工具执行库。

PreCog 会观察模型流式输出的工具调用事件。当它看到一个只读工具调用
以及对应的流式参数后，就可以在模型仍然生成后续内容时提前执行这个工具。
等真正的工具执行钩子到达时，如果实际的 `(tool, args)` 与缓存命中，
PreCog 会直接返回提前执行好的结果。

可以把它理解成把 CPU 分支预测、LLM speculative decoding 这类延迟优化手段，
搬到了 agent 的工具调用层：

| 层级 | 推测手段 | 猜错时浪费什么 | 猜中时节省什么 |
| --- | --- | --- | --- |
| CPU | 分支预测 | 被丢弃的流水线 | 吞吐 |
| LLM runtime | draft tokens | 被丢弃的 token | 解码延迟 |
| PreCog | 提前执行安全工具调用 | 一次只读工具调用 | agent 总耗时 |

## 状态

Alpha runtime。当前仓库是独立 Python 项目，不依赖 Node 或 pnpm。

已实现：

- 只读工具 allowlist 的幂等性注册表
- 基于稳定 JSON 的精确参数 cache key
- 带 TTL/LRU 的推测结果缓存
- token-Jaccard 模糊命中回退
- 带最近参数记忆的 bigram 工具预测器
- 工具名刚出现时就用最近参数提前推测
- 参数流结束后发现猜错时取消错误的进行中任务
- 命中率过低时自适应暂停新的推测
- 进行中推测数量上限，避免抢占真实工具调用资源
- 本地 `ToolRegistry`，可以直接注册和执行真实 Python 函数
- 可选 OpenAI Responses 与 LangGraph 适配器
- JSONL trace 与 predictor 状态持久化
- 更完整的运行时统计：resolved、cancelled、wasted、paused
- 异步推测执行器
- `before_execute` / `after_execute` 集成钩子
- smoke demo、合成 benchmark 和 unittest 测试套件

## 安装

```bash
python -m pip install -e .
```

如果只是本地开发，不安装也可以直接运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python demo/smoke.py
PYTHONPATH=src python demo/registry_quickstart.py
PYTHONPATH=src python demo/benchmark.py
```

## 快速开始

```python
import asyncio
import json
from typing import Any, Mapping

from precog import PreCog


async def tool_executor(tool_name: str, args: Mapping[str, Any]) -> Any:
    # 在这里调用你自己的真实工具执行逻辑。
    await asyncio.sleep(0.2)
    return {"tool": tool_name, "args": dict(args)}


async def main() -> None:
    precog = PreCog(
        read_only_tools={"search", "fetch_url", "read_file"},
        executor=tool_executor,
    )

    call_id = "call-1"
    args = {"q": "agent runtime architecture"}

    await precog.observe_model_event(
        {"type": "tool_call_start", "callId": call_id, "toolName": "search"}
    )
    await precog.observe_model_event(
        {"type": "tool_call_args_delta", "callId": call_id, "delta": json.dumps(args)}
    )
    await precog.observe_model_event({"type": "tool_call_end", "callId": call_id})

    decision = await precog.before_execute("search", args, call_id=call_id)
    if decision.type == "provide_result":
        result = decision.result
    else:
        result = await tool_executor("search", args)
        await precog.after_execute("search", args, result, call_id=call_id)

    print(result)


asyncio.run(main())
```

如果你的集成环境没有独立的 before/after 钩子，也可以用便捷封装：

```python
result = await precog.execute("search", {"q": "python agents"}, tool_executor)
```

## 真实工具注册表

独立 Python 应用可以直接用 `ToolRegistry` 做工具分发：

```python
from precog import IdempotencyClass, PreCog, ToolRegistry

registry = ToolRegistry()


@registry.register(idempotency_class=IdempotencyClass.NETWORK_READ, timeout_seconds=3)
def search(q: str) -> dict[str, str]:
    return {"query": q}


precog = PreCog(**registry.precog_kwargs())
result = await precog.execute("search", {"q": "agent latency"}, registry.execute)
```

同步函数默认会放进 worker thread 执行，避免推测执行阻塞事件循环。

## 运行时控制

默认配置足够激进，可以更早抢跑工具调用，同时保留保护栏：

```python
precog = PreCog(
    read_only_tools={"search", "fetch_url"},
    executor=tool_executor,
    predict_on_tool_start=True,
    adaptive_min_calls=20,
    adaptive_min_hit_rate=0.2,
    adaptive_cooldown_seconds=30,
    max_concurrent_speculations=8,
)
```

- `predict_on_tool_start` 会在模型刚输出工具名时就开始推测，参数使用该工具最近一次观察到的参数。
- 如果后续流式参数与猜测参数不一致，错误的进行中任务会被取消，并计入 wasted speculation。
- 自适应保护会在观测命中率低于 `adaptive_min_hit_rate` 时临时暂停新的推测。
- `max_concurrent_speculations` 限制同时进行的推测任务数量，避免无限抢占资源。

## 集成

### OpenAI Responses streaming

```python
from precog.adapters.openai import OpenAIResponsesAdapter

adapter = OpenAIResponsesAdapter()

for event in stream:
    for model_event in adapter.events_from(event):
        await precog.observe_model_event(model_event)
```

这个适配器会把 function-call 参数流转换成 PreCog 的通用 `ModelEvent`。

### LangGraph ToolNode

```python
from langgraph.prebuilt import ToolNode
from precog.adapters.langgraph import make_langgraph_tool_wrappers

tool_node = ToolNode(
    tools,
    **make_langgraph_tool_wrappers(precog),
)
```

wrapper 会先调用 `before_execute`，命中时直接返回缓存的 ToolNode 结果；
未命中时正常执行工具，并在 `after_execute` 中写入缓存。

## 运维

```python
from precog import JsonlTraceSink

precog = PreCog(
    **registry.precog_kwargs(),
    trace_sink=JsonlTraceSink("logs/precog.jsonl"),
)

precog.save_state("precog-state.json")
precog.load_state("precog-state.json")
await precog.close(cancel=True)
```

这一版的调研依据和设计取舍见 [research notes](docs/research-notes.md)。

## 安全模型

推测执行必须显式开启。默认情况下，所有工具都被视为不安全，只有你标记为
只读的工具才会被提前执行：

```python
precog = PreCog(read_only_tools={"search", "fetch_url"})
```

你也可以显式声明每个工具的幂等性类别：

```python
precog = PreCog(
    idempotency_classes={
        "search": "network_read",
        "write_file": "local_write_idempotent",
        "send_email": "remote_write",
    }
)
```

只有 `pure_read` 和 `network_read` 会被允许推测执行。会修改状态的工具不会命中
推测缓存，会照常真实执行。

## Benchmark

demo benchmark 模拟了 12 次工具调用，每次调用都有模型流式输出时间、后续思考
时间和真实工具延迟：

```bash
PYTHONPATH=src python demo/benchmark.py
```

它会比较三种模式：

1. 不启用 PreCog 的 baseline
2. 只使用 memoization 的缓存命中
3. memoization 加流式阶段的推测执行

具体数字会受到机器和事件循环调度影响，但整体形态应该是：当只读工具调用重复
或可以被提前执行时，真实工具执行次数减少，总 wall time 下降。

## 推到 GitHub

当前项目可以作为独立 Python 仓库发布：

```bash
git add .
git commit -m "Rewrite PreCog as a Python package"
git branch -M main
git remote add origin git@github.com:<you>/precog.git
git push -u origin main
```

把 `<you>` 换成你的 GitHub 用户名或组织名即可。

## License

MIT
