import React, { useState } from "react";
import { predictSentiment } from "../api";
import "./Sentiment.css";

export default function SentimentBox() {
  const [text, setText] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleCheck = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult("");

    try {
      const res = await predictSentiment(text);
      setResult(res.data.sentiment);
    } catch (err) {
      setResult("Server error...");
    }

    setLoading(false);
  };

  return (
    <div className="main-container">
      <div className="card">
        <h1 className="title">AI Sentiment Analyzer</h1>
        <p className="subtitle">Analyze emotions using Machine Learning</p>

        <textarea
          className="textbox"
          placeholder="Type your sentence here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button className="analyze-btn" onClick={handleCheck}>
          {loading ? "Analyzing..." : "Analyze Sentiment"}
        </button>

        {result && (
          <div className="result-box">
            <span>Result:</span> {result}
          </div>
        )}
      </div>
    </div>
  );
}
