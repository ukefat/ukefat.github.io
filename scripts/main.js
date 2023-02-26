// const myHeading = document.querySelector("h1");
// myHeading.textContent = "Hello world!";

// document.querySelector("h1").addEventListener("click", function () {
//     alert("Ouch! Stop poking me!");
//   });

const myImage = document.querySelector("img");

myImage.onclick = () => {
  const mySrc = myImage.getAttribute("src");
  if (mySrc === "images/comp-cat.jpg") {
    myImage.setAttribute("src", "images/comp-mike.png");
  }
  else {
    myImage.setAttribute("src", "images/comp-cat.jpg");
  }
};

let myButton = document.querySelector("button");
let myHeading = document.querySelector("h1");

function setUserName() {
  const myName = prompt("Please enter your name.");
  if (!myName) {
    setUserName();
  } else {
    localStorage.setItem("name", myName);
    myHeading.textContent = `Welcome, ${myName}`;
  }
}

if (!localStorage.getItem("name")) {
  setUserName();
} else {
  const storedName = localStorage.getItem("name");
  myHeading.textContent = `I know you, you are ${storedName}`;
}

myButton.onclick = () => {
  setUserName();
};