export const LETTER_CHECKPOINTS = {
  A: [
    { x: 50, y: 100 },
    { x: 25, y: 50 },
    { x: 75, y: 50 },
    { x: 35, y: 75 },
    { x: 65, y: 75 }
  ],
  B: [
    { x: 25, y: 100 },
    { x: 25, y: 50 },
    { x: 60, y: 50 },
    { x: 60, y: 75 },
    { x: 25, y: 75 }
  ],
  // Add more letters as needed
  default: [
    { x: 50, y: 50 },
    { x: 50, y: 100 }
  ]
};

export const getLetterCheckpoints = (letter) => {
  return LETTER_CHECKPOINTS[letter.toUpperCase()] || LETTER_CHECKPOINTS.default;
};