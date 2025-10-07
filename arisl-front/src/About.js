import React from "react";

export default function About() {
  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white shadow-lg rounded-xl p-8">
      <h2 className="text-indigo-700 text-3xl font-bold mb-4 font-serif">
        About ARISL Project
      </h2>
      <p className="text-lg text-gray-700 mb-4 leading-relaxed font-sans">
        ARISL is a cutting-edge AI-powered Indian Sign Language interpreter designed to empower communication through real-time gesture recognition. Combining
        advanced computer vision with deep learning models, the system translates hand gestures captured from your webcam into text seamlessly and instantly.
      </p>
      <ul className="list-disc list-inside text-indigo-700 font-semibold space-y-1 font-mono">
        <li>Built with TensorFlow, MediaPipe, FastAPI, and React</li>
        <li>Privacy friendly: all gesture analysis runs locally</li>
        <li>Accessible, user-friendly interface for all skill levels</li>
      </ul>
    </div>
  );
}
