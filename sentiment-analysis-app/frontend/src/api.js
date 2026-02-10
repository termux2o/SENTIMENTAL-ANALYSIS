import axios from "axios";

// load from .env
const API = axios.create({
  baseURL: process.env.REACT_APP_BACKEND_API_LINK
});

export const predictSentiment = (text) =>
  API.post("/predict", { text });
