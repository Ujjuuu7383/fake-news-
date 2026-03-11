async function checkFact() {
  const claim = document.getElementById("claimInput").value.trim();
  if (!claim) return alert("Enter a claim");

  document.getElementById("result").classList.add("hidden");

  const res = await fetch("/factcheck", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ claim })
  });

  const data = await res.json();

  const verdictBox = document.getElementById("verdictBox");
  verdictBox.className = "";

  verdictBox.innerHTML = `
    ${data.final_verdict} <br/>
    Confidence: ${(data.confidence * 100).toFixed(1)}%
  `;

  verdictBox.classList.add(
    data.final_verdict === "REAL" ? "verdict-real" :
    data.final_verdict === "FAKE" ? "verdict-fake" :
    "verdict-unknown"
  );

  const analysesDiv = document.getElementById("analyses");
  analysesDiv.innerHTML = "";

  data.analyses.forEach(a => {
    const div = document.createElement("div");
    div.className = "article";
    div.innerHTML = `
      <h4>${a.title}</h4>
      <p>${a.reasoning}</p>
      <span class="badge ${a.relationship}">${a.relationship}</span>
    `;
    analysesDiv.appendChild(div);
  });

  document.getElementById("result").classList.remove("hidden");
}
