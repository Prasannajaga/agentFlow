import { Agent } from "@cursor/sdk";

function emit(event) {
  process.stdout.write(`${safeJsonStringify(event)}\n`);
}

function safeJsonStringify(value) {
  const seen = new WeakSet();
  return JSON.stringify(value, (_key, current) => {
    if (typeof current === "bigint") {
      return current.toString();
    }
    if (current instanceof Error) {
      return {
        name: current.name,
        message: current.message,
        stack: current.stack,
      };
    }
    if (current && typeof current === "object") {
      if (seen.has(current)) {
        return "[Circular]";
      }
      seen.add(current);
    }
    return current;
  });
}

function assertString(value, fieldName) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${fieldName} must be a non-empty string.`);
  }
  return value;
}

async function readStdinJson() {
  let input = "";
  process.stdin.setEncoding("utf8");
  for await (const chunk of process.stdin) {
    input += chunk;
  }

  if (!input.trim()) {
    throw new Error("Bridge stdin payload is empty.");
  }

  try {
    return JSON.parse(input);
  } catch (error) {
    throw new Error(`Bridge received invalid JSON payload: ${error.message}`);
  }
}

async function main() {
  const request = await readStdinJson();

  const apiKey = assertString(request.apiKey, "apiKey");
  const cwd = assertString(request.cwd, "cwd");
  const prompt = assertString(request.prompt, "prompt");
  const model = typeof request.model === "string" && request.model.trim() ? request.model.trim() : "auto";

  emit({ type: "bridge_started", payload: { cwd } });

  const agentOptions = {
    apiKey,
    local: { cwd },
  };

  if (model !== "auto") {
    agentOptions.model = { id: model };
  }

  const agent = await Agent.create(agentOptions);

  let run;
  const metadata = request.metadata && typeof request.metadata === "object" ? request.metadata : undefined;
  try {
    run = metadata ? await agent.send(prompt, { metadata }) : await agent.send(prompt);
  } catch (error) {
    // Backward-compatible fallback for SDKs that don't support options in send().
    run = await agent.send(prompt);
  }

  const stream = typeof run?.stream === "function" ? run.stream() : run;

  for await (const event of stream) {
    emit({ type: "cursor_event", payload: event });
  }

  emit({ type: "cursor_completed", payload: { status: "completed" } });
  emit({ type: "bridge_completed", payload: {} });
}

main()
  .then(() => {
    process.exitCode = 0;
  })
  .catch((error) => {
    emit({
      type: "cursor_failed",
      payload: {
        message: error?.message || "Cursor bridge failed.",
        name: error?.name || "Error",
      },
    });

    const message = error?.stack || error?.message || String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
