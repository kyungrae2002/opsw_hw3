const $ = (id) => document.getElementById(id);
const text = $("text");

function updateCount() {
  $("count").textContent = `${text.value.length} / 5000`;
}
text.addEventListener("input", updateCount);

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    text.value = chip.dataset.text;
    updateCount();
    analyze();
  });
});

$("go").addEventListener("click", analyze);
text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze();
});

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = false;
}

async function analyze() {
  $("error").hidden = true;
  if (!text.value.trim()) {
    showError("분석할 문장을 입력해 주세요.");
    return;
  }
  $("go").disabled = true;
  $("go").textContent = "분석 중...";
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text.value }),
    });
    const data = await res.json();
    if (!res.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "요청 형식이 올바르지 않습니다.";
      throw new Error(detail);
    }
    render(data);
  } catch (err) {
    showError(err.message || "서버에 연결할 수 없습니다.");
  } finally {
    $("go").disabled = false;
    $("go").textContent = "분석하기";
  }
}

function pct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function render(d) {
  const pos = d.label === "positive";
  $("result").hidden = false;
  $("result").className = `card ${pos ? "is-pos" : "is-neg"}`;
  $("emoji").textContent = pos ? "😊" : "😞";
  $("label").textContent = pos ? "긍정적인 리뷰" : "부정적인 리뷰";
  $("score").textContent = `확신도 ${pct(d.score)}`;
  $("bar-neg").style.width = pct(d.scores.negative);
  $("bar-pos").style.width = pct(d.scores.positive);

  const ul = $("detail");
  ul.innerHTML = "";
  d.detail.forEach((c) => {
    const li = document.createElement("li");
    li.innerHTML = `<span></span><span class="mini"><i></i></span><span></span>`;
    li.children[0].textContent = c.label;
    li.children[1].firstChild.style.width = pct(c.probability);
    li.children[2].textContent = pct(c.probability);
    ul.appendChild(li);
  });
  $("meta").textContent =
    `모델 ${d.model} · ${d.tokens}토큰${d.truncated ? "(길어서 일부 잘림)" : ""} · ${d.inference_ms}ms · ` +
    `서버 v${d.version} (${d.commit})`;
}

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    $("server").textContent = h.model.loaded
      ? `서버 v${h.version} (${h.commit}) · 모델 ${h.model.name}`
      : `모델 로드 실패: ${h.model.error}`;
  })
  .catch(() => ($("server").textContent = "서버에 연결할 수 없습니다."));
