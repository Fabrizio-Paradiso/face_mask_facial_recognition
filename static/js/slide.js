/* Slide Title */
const texts = document.querySelector(".animate-text").children
const textsLength = texts.length
let index = 0
const textInTimer = 3000
const textOutTimer = 2800

function animateText() {
  for(let i=0; i<textsLength; i++){
    texts[i].classList.remove("text-in","text-out");  
  }
  texts[index].classList.add("text-in");

  setTimeout(function(){
    texts[index].classList.add("text-out");              
  },textOutTimer)

  setTimeout(function(){

  if(index == textsLength-1){
      index=0;
  }
  else{
    index++;
  }
  animateText();
  },textInTimer); 
}

window.onload=animateText;