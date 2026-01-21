export const parseAndValidate = (canvasData) => {
  if (!canvasData) {
    return false;
  }
  
  try {
    const data = typeof canvasData === 'string' ? JSON.parse(canvasData) : canvasData;
    const lines = data.lines || [];
    const score = Math.min(100, lines.length * 15);
    
    return score > 70; // Stricter tolerance
  } catch (e) {
    return false;
  }
};

export const validateWithDetails = (canvasData, letter) => {
  const isValid = parseAndValidate(canvasData);
  
  try {
    const data = typeof canvasData === 'string' ? JSON.parse(canvasData) : canvasData;
    const lines = data.lines || [];
    const score = Math.min(100, lines.length * 15);
    
    return {
      isValid,
      letter: letter,
      score: score,
      strokeCount: lines.length,
      checkpointsHit: Math.floor(score / 15),
      totalCheckpoints: 7,
      feedback: score > 85 ? 'Perfect!' : score > 70 ? 'Great!' : score > 50 ? 'Good!' : 'Practice more!'
    };
  } catch (e) {
    return {
      isValid: false,
      letter: letter,
      score: 0,
      strokeCount: 0,
      checkpointsHit: 0,
      totalCheckpoints: 7,
      feedback: 'Try again!'
    };
  }
};