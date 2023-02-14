/* Get People List */
let canvas = document.getElementById("empty-canvas")
let context = canvas.getContext("2d")
canvas.style.display = "none"

let json = (function() {
  let json = null;
  $.ajax({
    'async': false,
    'global': false,
    'url': "static/json/people.json",
    'dataType': "json",
    'success': function(data) {
      json = data;
    }
  });
  return json;
})();

const $people_button = document.querySelector("#people-button")
$people_button.addEventListener("click", async () => {
  let pdf = new jsPDF("portrait", "mm", "a4")
  let urlSources = []
  setFontAndTitlePDF(pdf)
  json.forEach(function(person, index){
    pdf.text(40, 35 + (index*25), `ID: ${person.id}`)
    pdf.text(40, 40 + (index*25), `First Name: ${person.name}`);
    urlSources.push(person.image)
  });
  await addImagesToPDFandSavePDF(urlSources, pdf)
});

async function addImagesToPDFandSavePDF(images, pdf){
  for(const [index, image] of enumerate(images)){
    await addImageToPDF(image, index, pdf)
  }
  pdf.save(`People_List.pdf`)
}

function addImageToPDF(url, index, pdf) {
  return new Promise(resolve => {
    let image = new Image();
    image.onload = (() => {
      drawImageScaled(image)
      let dataUrl = canvas.toDataURL()
      pdf.addImage(dataUrl, "png", 15, 5 + index*25, 25, 15)
      resolve(pdf)
    });
    image.src = url;
  });
}

function drawImageScaled(image) {
  var widthRatio = canvas.width / image.width
  var heightRatio =  canvas.height / image.height
  var ratio  = Math.min(widthRatio, heightRatio)
  var centerShiftX = (canvas.width - image.width*ratio ) / 2
  var centerShiftY = (canvas.height - image.height*ratio ) / 2
  context.clearRect(0, 0, canvas.width, canvas.height)
  context.drawImage(image, 0, 0, image.width, image.height, centerShiftX, centerShiftY, image.width*ratio, image.height*ratio)
}

function* enumerate(iterable) {
  let i = 1;
  for (const x of iterable) {
    yield [i, x];
    i++;
  }
}

function setFontAndTitlePDF(pdf){
  pdf.setFont("helvetica");
  pdf.setFontSize(9);
  pdf.text(95, 20, "People List")
}
