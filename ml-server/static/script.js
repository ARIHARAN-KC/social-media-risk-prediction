async function analyzeText() {
    const text = document.getElementById("textInput").value;

    if (!text.trim()) {
        alert("Please enter text.");
        return;
    }

    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (data.error) {
        alert(data.error);
        return;
    }

    document.getElementById("result").classList.remove("hidden");

    document.getElementById("label").innerText =
        "Risk Level: " + data.label;

    document.getElementById("confidence").innerText =
        "Confidence: " + (data.confidence * 100).toFixed(2) + "%";

    document.getElementById("lowBar").style.width =
        (data.probabilities["Low Risk"] * 100) + "%";

    document.getElementById("mediumBar").style.width =
        (data.probabilities["Medium Risk"] * 100) + "%";

    document.getElementById("highBar").style.width =
        (data.probabilities["High Risk"] * 100) + "%";

    document.getElementById("lowBar").innerText =
        "Low " + (data.probabilities["Low Risk"] * 100).toFixed(1) + "%";

    document.getElementById("mediumBar").innerText =
        "Medium " + (data.probabilities["Medium Risk"] * 100).toFixed(1) + "%";

    document.getElementById("highBar").innerText =
        "High " + (data.probabilities["High Risk"] * 100).toFixed(1) + "%";
}