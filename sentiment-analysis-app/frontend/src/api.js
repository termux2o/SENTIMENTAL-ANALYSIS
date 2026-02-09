import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000"
});

export const predictSentiment = (text) =>
  API.post("/predict", { text });
