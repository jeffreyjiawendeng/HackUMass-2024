// Define each sentence as an array of strings
const sentences = [
  "This program uses machine learning to capture and analyze facial expressions via your device's front camera to identify emotions like happiness, sadness, and anger.",
  "User privacy is a top priority—no images are stored or shared.",
  "This technology can enhance applications in areas such as customer service, mental health monitoring, and education, while ensuring ethical standards and respect for user privacy."
];

let sentenceIndex = 0;
let charIndex = 0;
const typingSpeed = 10;       // Typing speed for each character
const sentenceDelay = 100;    // Delay between sentences

function typeSentence() {
  if (sentenceIndex < sentences.length) {
    const currentSentence = sentences[sentenceIndex];
    if (charIndex < currentSentence.length) {
      document.getElementById("typedText").innerHTML += currentSentence.charAt(charIndex);
      charIndex++;
      setTimeout(typeSentence, typingSpeed);
    } else {
      // Move to the next sentence after a delay
      sentenceIndex++;
      charIndex = 0;
      document.getElementById("typedText").innerHTML += "<br><br>"; // Add line break after each sentence
      setTimeout(typeSentence, sentenceDelay);
    }
  }
}

window.onload = typeSentence; // Start typing effect on page load
