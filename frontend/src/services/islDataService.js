// ISL Data Service - manages sign images and data
class ISLDataService {
  constructor() {
    // Base path for ISL sign images (update this when you add actual images)
    this.basePath = '/src/assets/isl-signs/';
    
    // Available letters and words from your AiModel/vectors directory
    this.availableLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];
    this.availableWords = ['Hello', 'Thankyou', 'Father', 'Mother', 'Husband', 'Wife'];
    this.availableNumbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'];
  }

  // Get image path for a sign (placeholder for now, will use actual images when available)
  getSignImagePath(sign) {
    // For now, return a placeholder or generate a simple SVG
    return this.generatePlaceholderImage(sign);
  }

  // Generate a simple placeholder image as data URL
  generatePlaceholderImage(sign) {
    const canvas = document.createElement('canvas');
    canvas.width = 200;
    canvas.height = 200;
    const ctx = canvas.getContext('2d');
    
    // Background
    ctx.fillStyle = '#f0f9ff';
    ctx.fillRect(0, 0, 200, 200);
    
    // Border
    ctx.strokeStyle = '#0ea5e9';
    ctx.lineWidth = 3;
    ctx.strokeRect(5, 5, 190, 190);
    
    // Sign text
    ctx.fillStyle = '#0369a1';
    ctx.font = 'bold 48px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(sign, 100, 80);
    
    // ISL label
    ctx.font = '16px Arial';
    ctx.fillText('ISL Sign', 100, 130);
    
    // Hand icon (simple representation)
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(100, 160, 15, 0, 2 * Math.PI);
    ctx.fill();
    
    return canvas.toDataURL();
  }

  // Get all available signs
  getAllSigns() {
    return {
      letters: this.availableLetters,
      words: this.availableWords,
      numbers: this.availableNumbers
    };
  }

  // Get random signs for quiz
  getRandomSigns(count = 5, type = 'letters') {
    const signs = type === 'letters' ? this.availableLetters : 
                  type === 'words' ? this.availableWords : 
                  this.availableNumbers;
    
    const shuffled = [...signs].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count).map(sign => ({
      sign,
      image: this.getSignImagePath(sign),
      type
    }));
  }

  // Get quiz questions with multiple choice options
  generateQuizQuestions(count = 5) {
    const questions = [];
    const usedSigns = new Set();
    
    while (questions.length < count) {
      // Mix of letters and words
      const useLetters = Math.random() > 0.3;
      const signs = useLetters ? this.availableLetters : this.availableWords;
      
      // Get a random correct answer
      const correctAnswer = signs[Math.floor(Math.random() * signs.length)];
      if (usedSigns.has(correctAnswer)) continue;
      usedSigns.add(correctAnswer);
      
      // Generate wrong options
      const wrongOptions = [];
      while (wrongOptions.length < 3) {
        const wrongSign = signs[Math.floor(Math.random() * signs.length)];
        if (wrongSign !== correctAnswer && !wrongOptions.includes(wrongSign)) {
          wrongOptions.push(wrongSign);
        }
      }
      
      // Shuffle options
      const options = [correctAnswer, ...wrongOptions].sort(() => 0.5 - Math.random());
      
      questions.push({
        id: questions.length + 1,
        image: this.getSignImagePath(correctAnswer),
        question: `What does this ISL sign represent?`,
        options,
        correctAnswer,
        type: useLetters ? 'letter' : 'word'
      });
    }
    
    return questions;
  }

  // Get flashcard data
  getFlashcardData(letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']) {
    return letters.map(letter => ({
      letter,
      image: this.getSignImagePath(letter),
      description: `ISL sign for letter ${letter}`,
      tips: this.getSignTips(letter)
    }));
  }

  // Get tips for each sign
  getSignTips(sign) {
    const tips = {
      'A': 'Make a fist with thumb on the side',
      'B': 'Four fingers up, thumb across palm',
      'C': 'Curved hand like holding a cup',
      'D': 'Index finger up, thumb and middle finger touch',
      'E': 'Fingers bent down touching thumb',
      'F': 'Index and thumb touch, other fingers up',
      'G': 'Index finger and thumb point sideways',
      'H': 'Index and middle finger sideways',
      'Hello': 'Wave your hand in a greeting motion',
      'Thankyou': 'Touch chin and move hand forward',
      'Father': 'Thumb touches forehead',
      'Mother': 'Thumb touches chin',
      'Husband': 'Sign for man then marriage',
      'Wife': 'Sign for woman then marriage',
      // Common Indian words
      'Apple': 'Make fist, then twist at wrist like picking apple from tree',
      'Banana': 'Peel imaginary banana with fingers',
      'Mango': 'Cup hand and bring to mouth like eating sweet fruit',
      'Rice': 'Scoop with hand like eating rice with fingers',
      'Chapati': 'Roll flat circle with both hands',
      'Milk': 'Squeeze fist like milking cow',
      'Tea': 'Sip from imaginary cup with pinky up',
      'Water': 'Tap chin with W-hand shape (thirst sign)',
      'Book': 'Open palms like opening book pages',
      'Pen': 'Write in air with finger',
      'Chair': 'Sit motion with hands on imaginary armrests',
      'Table': 'Flat surface with both hands, then legs',
      'School': 'Clap hands like teacher getting attention',
      'Home': 'Touch corner of mouth, then make roof shape',
      'Family': 'F-hands in circle motion around each other',
      'Friend': 'Hook index fingers together',
      'Happy': 'Brush chest upward with both hands',
      'Sad': 'Trace tears down cheeks with fingers',
      'Good': 'Touch lips then move hand forward',
      'Bad': 'Touch lips then flip hand down',
      'Yes': 'Nod fist up and down',
      'No': 'Shake index and middle finger side to side',
      'Please': 'Rub chest in circular motion',
      'Sorry': 'Rub fist on chest in circular motion',
      'Help': 'Lift one hand with the other',
      'Work': 'Hit fist on top of other fist',
      'Play': 'Shake Y-hands back and forth',
      'Eat': 'Bring fingers to mouth repeatedly',
      'Drink': 'Tilt imaginary glass to mouth',
      'Sleep': 'Rest head on hand like pillow',
      'Walk': 'Two fingers walk on palm',
      'Run': 'Hook thumbs and wiggle fingers while moving forward',
      'Stop': 'Flat hand hits other palm',
      'Go': 'Point and move hand forward',
      'Come': 'Wave hand toward yourself',
      'Big': 'Hands far apart showing size',
      'Small': 'Pinch fingers close together',
      'Hot': 'Quickly pull hand away from mouth',
      'Cold': 'Shiver and hug yourself',
      'Beautiful': 'Circle face with hand, then open fingers',
      'Ugly': 'Hook index finger and pull across nose'
    };
    
    return tips[sign] || `Practice the ${sign} sign slowly and clearly`;
  }

  // Check if actual image files exist (for future use)
  async checkImageExists(imagePath) {
    try {
      const response = await fetch(imagePath);
      return response.ok;
    } catch {
      return false;
    }
  }

  // Future method: Load actual images from your dataset
  async loadActualImages() {
    // This will be implemented when you add actual image files
    // It will scan the assets folder and map real images to signs
    console.log('Actual image loading will be implemented when image files are added');
  }
}

export default new ISLDataService();