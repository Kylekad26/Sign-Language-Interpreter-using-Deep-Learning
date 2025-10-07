import React, { useState, useRef, useEffect } from "react";
import { loadModel, predictAction } from "./ModelLoader";
import Webcam from "react-webcam";
import {
  VideoCameraIcon,
  SparklesIcon,
  ArrowPathIcon,
  ClipboardDocumentIcon,
  SpeakerWaveIcon,
  ChatBubbleLeftIcon,
} from "@heroicons/react/24/outline";

const themes = {
  cyberpunk: {
    bg: "bg-gradient-to-tr from-purple-900 via-pink-700 to-red-600",
    text: "text-pink-400",
    panelBg: "bg-black/90",
    accent: "bg-pink-600 hover:bg-pink-700",
    btnText: "text-white",
    font: "'Roboto Mono', monospace",
  },
  matrix: {
    bg: "bg-gradient-to-b from-black via-green-900 to-black",
    text: "text-green-400",
    panelBg: "bg-black/95",
    accent: "bg-green-700 hover:bg-green-800",
    btnText: "text-green-100",
    font: "'Source Code Pro', monospace",
  },
  futuristic: {
    bg: "bg-gradient-to-tr from-blue-900 via-cyan-700 to-indigo-700",
    text: "text-cyan-300",
    panelBg: "bg-gray-900/90",
    accent: "bg-cyan-600 hover:bg-cyan-700",
    btnText: "text-white",
    font: "'Space Mono', monospace",
  },
};

export default function SignLanguageMain() {
  const [model, setModel] = useState(null);
  const [recording, setRecording] = useState(false);
  const [sequence, setSequence] = useState([]);
  const [words, setWords] = useState([]);
  const [sentence, setSentence] = useState("");
  const [status, setStatus] = useState("Ready to start? 🔧");
  const [loadingModel, setLoadingModel] = useState(false);
  const [themeName, setThemeName] = useState("cyberpunk");
  const webcamRef = useRef(null);
  const countdownTimerRef = useRef(null);
  const [countdown, setCountdown] = useState(null);

  const theme = themes[themeName];

  useEffect(() => {
    document.title = "ARISL - Techy Sign Language Demo";
  }, []);

  useEffect(() => {
    if (countdown === 0) {
      setCountdown(null);
      handleRecord();
    }
  }, [countdown]);

  useEffect(() => {
    return () => {
      if (countdownTimerRef.current) {
        clearInterval(countdownTimerRef.current);
      }
    };
  }, []);

  const handleStart = async () => {
    setLoadingModel(true);
    setStatus("Booting AI model... 🔌");
    const loaded = await loadModel();
    if (loaded) {
      setModel(loaded);
      setStatus("Model online 🚀 Ready to decode gestures 🔍");
    } else {
      setStatus("Model failed to load ❌ Please refresh");
    }
    setLoadingModel(false);
  };

  const startCountdownAndRecord = () => {
    if (!model) return setStatus("Model not ready! ⚠️");
    let c = 3;
    setCountdown(c);
    setStatus("Calibrating... Get ready 🔥");
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    countdownTimerRef.current = setInterval(() => {
      c--;
      if (c < 0) {
        clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      } else {
        setCountdown(c);
      }
    }, 1000);
  };

  const handleRecord = () => {
    setRecording(true);
    setSequence([]);
    setStatus("Recording gestures ⚡");
    let frames = [];
    let frameCount = 0;
    const interval = setInterval(async () => {
      if (!webcamRef.current) {
        clearInterval(interval);
        setRecording(false);
        setStatus("Webcam not detected! ❌");
        return;
      }
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const blob = await fetch(imageSrc).then((res) => res.blob());
        frames.push(blob);
        setSequence([...frames]);
        frameCount++;
        setStatus(`Captured frame ${frameCount}/30 🖼️`);
        if (frames.length === 30) {
          clearInterval(interval);
          setRecording(false);
          analyzeGesture(frames);
        }
      }
    }, 80);
  };

  const analyzeGesture = async (frames) => {
    setStatus("Processing... AI analyzing 🤖");
    const result = await predictAction(frames);
    if (result && result.confidence > 0.45) {
      setWords((prev) => [...prev, result.predicted_word]);
      setSentence((prev) =>
        prev ? `${prev} ${result.predicted_word}` : result.predicted_word
      );
      setStatus(`Detected: ${result.predicted_word} (${(result.confidence * 100).toFixed(1)}%) ✅`);
    } else {
      setStatus("Cannot recognize that gesture, try again ⚠️");
    }
  };

  const handleReset = () => {
    setWords([]);
    setSentence("");
    setStatus("System reset, ready to go 🔄");
  };

  const copySentence = () => {
    if (!sentence) return;
    navigator.clipboard.writeText(sentence);
    alert("Copied to clipboard! 📋");
  };

  const speakSentence = () => {
    if (!sentence) return;
    const utterance = new SpeechSynthesisUtterance(sentence);
    utterance.lang = "en-IN";
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div
      className={`${theme.bg} ${theme.text} min-h-screen p-6 transition-colors duration-700 font-mono`}
      style={{ fontFamily: theme.font }}
    >
      <header className="max-w-5xl mx-auto flex justify-between items-center mb-6 select-none">
        <h1 className="text-4xl font-extrabold tracking-widest uppercase drop-shadow-lg">
          ARISL <span className="text-white/80">Sign Language Interpreter</span>
        </h1>

        <select
          aria-label="Select Theme"
          value={themeName}
          onChange={(e) => setThemeName(e.target.value)}
          className="bg-black/70 text-white px-4 py-2 rounded-lg cursor-pointer font-semibold"
        >
          <option value="cyberpunk">Cyberpunk</option>
          <option value="matrix">Matrix</option>
          <option value="futuristic">Futuristic Blue</option>
        </select>
      </header>

      <main className={`${theme.panelBg} max-w-5xl mx-auto p-8 rounded-none shadow-none space-y-8`}>
        <p className="text-center text-2xl font-bold tracking-wide">{status}</p>

        {/* Webcam container */}
        <div className="max-w-4xl mx-auto border border-white/30">
          <Webcam
            ref={webcamRef}
            width={"100%"}
            height={540}
            screenshotFormat="image/jpeg"
            videoConstraints={{ facingMode: "user" }}
            className="block w-full object-contain"
          />
        </div>

        {/* Controls */}
        <div className="flex justify-center flex-wrap gap-6">
          <button
            onClick={handleStart}
            disabled={loadingModel || !!model}
            className={`${theme.accent} ${theme.btnText} flex gap-3 items-center rounded-none px-8 py-3 font-bold text-xl shadow-none transition duration-300`}
          >
            <VideoCameraIcon className="h-7 w-7" />
            {loadingModel ? "Loading..." : model ? "Model Ready" : "Start Model"}
          </button>
          <button
            onClick={() => {
              if (!recording) startCountdownAndRecord();
            }}
            disabled={!model || recording}
            className={`${theme.accent} ${theme.btnText} flex gap-3 items-center rounded-none px-8 py-3 font-bold text-xl shadow-none transition duration-300`}
          >
            <SparklesIcon className="h-7 w-7" />
            {recording ? "Recording..." : "Record Gesture"}
          </button>
          <button
            onClick={handleReset}
            className="bg-red-700 text-white flex gap-3 items-center rounded-none px-8 py-3 font-bold text-xl shadow-none hover:bg-red-800 transition duration-300"
          >
            <ArrowPathIcon className="h-7 w-7" />
            Reset
          </button>
        </div>

        {countdown !== null && countdown >= 0 && (
          <div className="text-8xl font-extrabold text-center select-none text-white drop-shadow-lg">
            {countdown > 0 ? countdown : "Go!"}
          </div>
        )}

        {recording && (
          <div className="w-full bg-white/20 rounded-full h-6 max-w-4xl mx-auto shadow-inner">
            <div
              className="h-6 bg-gradient-to-r from-pink-600 via-purple-700 to-indigo-600 transition-all ease-in-out duration-300 rounded-full"
              style={{ width: `${(sequence.length / 30) * 100}%` }}
            />
          </div>
        )}

        <section className="max-w-4xl mx-auto bg-black/30 rounded-none p-6 shadow-none space-y-4 border border-white/20">
          <h2 className="text-white text-3xl font-extrabold tracking-wide">
            Words Detected ✨
          </h2>
          {words.length === 0 ? (
            <p className="text-green-400 italic">No words detected yet...</p>
          ) : (
            <div className="flex flex-wrap gap-4">
              {words.map((w, i) => (
                <span
                  key={i}
                  className="select-text cursor-pointer px-5 py-2 rounded bg-gradient-to-tr from-green-600 to-green-900 text-white font-semibold shadow-md hover:scale-110 transform transition"
                  title="Click to copy word"
                  onClick={() => {
                    navigator.clipboard.writeText(w);
                    alert(`Copied word: ${w}`);
                  }}
                >
                  {w}
                </span>
              ))}
            </div>
          )}
        </section>

        <section className="max-w-4xl mx-auto bg-gradient-to-r from-indigo-900 via-purple-900 to-pink-900 rounded-none p-6 shadow-none space-y-6 border border-white/30">
          <h2 className="text-cyan-300 text-3xl font-extrabold tracking-wide flex items-center gap-3">
            <ChatBubbleLeftIcon className="w-10 h-10 text-green-400" />
            Sentence
          </h2>
          <p
            className="text-green-300 font-mono text-3xl select-text break-words p-4 rounded bg-black/30 shadow-none"
            aria-live="polite"
          >
            {sentence || "No sentence formed yet! Try recording a gesture."}
          </p>
          <div className="flex gap-6">
            {sentence && (
              <>
                <button
                  onClick={copySentence}
                  className={`${theme.accent} ${theme.btnText} flex gap-2 items-center rounded px-6 py-2 font-semibold shadow-none`}
                >
                  <ClipboardDocumentIcon className="w-6 h-6" />
                  Copy Sentence
                </button>
                <button
                  onClick={speakSentence}
                  className={`${theme.accent} ${theme.btnText} flex gap-2 items-center rounded px-6 py-2 font-semibold shadow-none`}
                >
                  <SpeakerWaveIcon className="w-6 h-6" />
                  Speak Sentence
                </button>
              </>
            )}
          </div>
        </section>

        <footer className="max-w-4xl mx-auto mt-12 text-center text-green-400 font-semibold text-lg select-none">
          💡 <em>Tip: Use a dark room or neon-lit background for best gesture detection!</em>
        </footer>
      </main>
    </div>
  );
}
