const elements = {
  downloadStatus: document.querySelector("#download-status"),
  vllmStatus: document.querySelector("#vllm-status"),
  modelSize: document.querySelector("#model-size"),
  activityLog: document.querySelector("#activity-log"),
  promptInput: document.querySelector("#prompt-input"),
  profileInput: document.querySelector("#profile-input"),
  maxNewTokensInput: document.querySelector("#max-new-tokens-input"),
  refreshStatusButton: document.querySelector("#refresh-status-button"),
  startVllmButton: document.querySelector("#start-vllm-button"),
  compareButton: document.querySelector("#compare-button"),
  vllmTtft: document.querySelector("#vllm-ttft"),
  vllmTps: document.querySelector("#vllm-tps"),
  vllmE2e: document.querySelector("#vllm-e2e"),
  vllmOutput: document.querySelector("#vllm-output"),
  leanstackPrefill: document.querySelector("#leanstack-prefill"),
  leanstackTps: document.querySelector("#leanstack-tps"),
  leanstackE2e: document.querySelector("#leanstack-e2e"),
  leanstackOutput: document.querySelector("#leanstack-output"),
  throughputRatio: document.querySelector("#throughput-ratio"),
  latencyRatio: document.querySelector("#latency-ratio"),
};

const OFFICIAL_PROFILE = "decode_64_256";

function setBusy(isBusy) {
  elements.refreshStatusButton.disabled = isBusy;
  elements.startVllmButton.disabled = isBusy;
  elements.compareButton.disabled = isBusy;
}

function formatSeconds(value) {
  return value == null ? "-" : `${value.toFixed(3)} s`;
}

function formatRate(value) {
  return value == null ? "-" : `${value.toFixed(2)} tok/s`;
}

function formatRatio(value) {
  return value == null ? "-" : `${value.toFixed(2)}x`;
}

function formatBytes(value) {
  if (!value) return "-";
  const gib = value / (1024 ** 3);
  return `${gib.toFixed(2)} GiB`;
}

function setStatusValue(node, text, mode = "muted") {
  node.textContent = text;
  node.classList.remove("muted", "good", "warn");
  node.classList.add(mode);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function refreshStatus() {
  setBusy(true);
  elements.activityLog.textContent = "Checking remote model and vLLM state…";
  try {
    const payload = await requestJson("/api/status");
    const status = payload.status;
    setStatusValue(
      elements.downloadStatus,
      status.download_complete ? "Complete" : status.fetch_processes.length ? "Downloading" : "Waiting",
      status.download_complete ? "good" : status.fetch_processes.length ? "warn" : "muted",
    );
    setStatusValue(
      elements.vllmStatus,
      status.vllm_ready ? "Ready" : "Not running",
      status.vllm_ready ? "good" : "warn",
    );
    elements.modelSize.textContent = formatBytes(status.model_size_bytes);
    elements.activityLog.textContent = !status.download_complete
      ? status.fetch_processes.length
        ? `Remote download in progress: ${status.fetch_processes[0]}`
        : "Remote checkpoint is not ready yet."
      : !status.pack_ready
        ? "Checkpoint is ready, but leanpack artifact is missing."
        : "Remote checkpoint and leanpack artifact are ready.";
  } catch (error) {
    elements.activityLog.textContent = String(error);
    setStatusValue(elements.downloadStatus, "Error", "warn");
    setStatusValue(elements.vllmStatus, "Error", "warn");
  } finally {
    setBusy(false);
  }
}

async function startVllm() {
  setBusy(true);
  elements.activityLog.textContent = "Starting remote vLLM…";
  try {
    await requestJson("/api/start-vllm", {method: "POST", body: "{}"});
    elements.activityLog.textContent = "Remote vLLM is ready.";
    await refreshStatus();
  } catch (error) {
    elements.activityLog.textContent = String(error);
    setBusy(false);
  }
}

function renderComparison(result) {
  const vllm = result.vllm;
  const leanstack = result.leanstack;
  const leanstackTimings = leanstack.timings || {};
  const leanstackThroughput = leanstack.throughput || {};

  elements.vllmTtft.textContent = formatSeconds(vllm.ttft_seconds);
  elements.vllmTps.textContent = formatRate(vllm.generated_tokens_per_second);
  elements.vllmE2e.textContent = formatRate(vllm.end_to_end_tokens_per_second);
  elements.vllmOutput.textContent = vllm.generated_text || "(empty)";

  elements.leanstackPrefill.textContent = formatSeconds(leanstackTimings.prefill_seconds);
  elements.leanstackTps.textContent = formatRate(leanstackThroughput.runtime_tokens_per_second);
  elements.leanstackE2e.textContent = formatRate(leanstackThroughput.full_loop_tokens_per_second);
  elements.leanstackOutput.textContent = leanstack.generated_text || "(empty)";

  elements.throughputRatio.textContent = formatRatio(result.delta.runtime_tokens_per_second_ratio);
  elements.latencyRatio.textContent = formatRatio(result.delta.prefill_to_vllm_ttft_ratio);
}

async function compare() {
  const prompt = elements.promptInput.value.trim();
  if (!prompt) {
    elements.activityLog.textContent = "Prompt must not be empty.";
    return;
  }
  const maxNewTokensRaw = elements.maxNewTokensInput.value.trim();
  const body = {
    prompt,
    profile: OFFICIAL_PROFILE,
  };
  if (maxNewTokensRaw) {
    body.max_new_tokens = Number(maxNewTokensRaw);
  }

  setBusy(true);
  elements.activityLog.textContent = "Running vLLM and leanstack on the remote box…";
  try {
    const payload = await requestJson("/api/compare", {
      method: "POST",
      body: JSON.stringify(body),
    });
    renderComparison(payload.result);
    elements.activityLog.textContent = "Comparison complete.";
  } catch (error) {
    elements.activityLog.textContent = String(error);
  } finally {
    setBusy(false);
  }
}

elements.refreshStatusButton.addEventListener("click", refreshStatus);
elements.startVllmButton.addEventListener("click", startVllm);
elements.compareButton.addEventListener("click", compare);

refreshStatus();
