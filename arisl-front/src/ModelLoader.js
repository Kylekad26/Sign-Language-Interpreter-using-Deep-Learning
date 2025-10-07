// src/ModelLoader.js

export async function loadModel() {
  // Not needed for FastAPI backend, but kept for UI symmetry
  return true;
}

export async function predictAction(sequence) {
  // sequence: Array of 30 blobs (frames)
  const formData = new FormData();
  sequence.forEach((frame, idx) => {
    formData.append("files", frame, `frame${idx}.jpg`);
  });
  try {
    const res = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  } catch (err) {
    return null;
  }
}
