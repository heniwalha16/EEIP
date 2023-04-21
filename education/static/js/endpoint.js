
    
const microphoneButton = document.getElementById("microphone-button");
const recognition = new webkitSpeechRecognition();
recognition.lang = 'en-USA';

microphoneButton.addEventListener("click", () => {
  recognition.start();
});

recognition.addEventListener("result", (event) => {
  const speechToText = event.results[0][0].transcript;
  console.log(speechToText);
  const textOutput = document.getElementById("problem");
  textOutput.value = speechToText;

  // Send speechToText to your Django API for text conversion

});function choosePhoto() {
  const input = document.getElementById('photo-input');
  input.click();
}

function extractText() {
  const input = document.getElementById('photo-input');
  const file = input.files[0];

  const formData = new FormData();
  formData.append('image', file);

  fetch('/extract_text/', {
    method: 'POST',
    body: formData
  })
  .then(response => response.text())
  .then(data => {
    console.log(data);
    const output = document.getElementById('problem');
    output.value = data;
  })
  .catch(error => console.error(error));
}  
function speak() {
  const text = document.getElementById("problem").value;
  console.log(text);
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  speechSynthesis.speak(utterance);
  
}
function chat_solution() {
  
  const userInput = document.getElementById("problem1").textContent;
  console.log(JSON.stringify({ user_input: userInput }));

  fetch('/problem_solution/', {
    method: 'POST',
    body: JSON.stringify({ user_input: userInput }),
    headers: {
      'Content-Type': 'application/json'
    }
  })
  .then(response => response.text())
  .then(solution => {
    console.log(solution);
    const solutionElement = document.getElementById('solution');
    solutionElement.textContent = solution;
  })
  .catch(error => console.error(error));
}