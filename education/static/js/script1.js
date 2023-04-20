
const home = document.querySelector(".home")
var mascot = document.getElementById("mascot");
function fly() {
  mascot.style.transform = "translate(320%, 60%) rotate(360deg) scale(3)";
}
setTimeout(function() {
    document.getElementById("mascot").style.display = "block";
    // Add code here to start working
  }, 4000);
  setTimeout(function() {
    document.getElementById("text_mascot").style.display = "block";
    // Add code here to start working
  }, 7000);  

setTimeout(fly, 5000);


var btn1 = document.getElementById('btn1')
var btn1value = document.getElementById('btn1').value;

btn1.onmouseover  = function () {
    btn1.value = "Register"
}
btn1.onmouseout  = function() {
    btn1.value = btn1value;
}
var btn2 = document.getElementById('btn2')
var btn2value = document.getElementById('btn2').value;

btn2.onmouseover  = function () {
    btn2.value = "Login"
}
btn2.onmouseout  = function() {
    btn2.value = btn2value;
}

