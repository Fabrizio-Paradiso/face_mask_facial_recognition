/* Take Screenshot */
let today = new Date()
const $screenshot_button = document.querySelector("#screenshot-button"),
$target = document.querySelector("#video");
$screenshot_button.addEventListener("click", () => {
    html2canvas($target) 
      .then(canvas => {
        console.log(Date.now())
        let link = document.createElement("a");
        link.download = `Screenshot_${getMonth()}-${getDay()}-${getYear()}_${getHours()}_${getMinutes()}_${getSeconds()}.png`;
        link.href = canvas.toDataURL();
        link.click();
      });
  });


function getDay(){ 
  day = today.getDate()
  if(day<10){day = `0${day}`} 
  return today.getDate() 
}

function getMonth(){
  month = today.getMonth()+1
  if(month<10){month = `0${month}`} 
  return month
}

function getYear(){ return today.getFullYear()}

function getHours() {return today.getHours()}

function getMinutes() {return today.getMinutes()}

function getSeconds() {return today.getSeconds()}