var result;
document.getElementById('fileIn').onchange = function (evt) {
    var tgt = evt.target || window.event.srcElement,
        files = tgt.files;

    // FileReader support
    
        var fr = new FileReader();
        fr.onload = function () {
            document.getElementById('outImage').src = fr.result;    
        }
        fr.readAsDataURL(files[0]);
}

function SubmitClick(){
    formData = new FormData(document.getElementById('request_form'))
    fetch('/api/upload_image',{method:'POST',body: formData}).then(response => response.json()).then(response => console.log(JSON.stringify(response)))
}