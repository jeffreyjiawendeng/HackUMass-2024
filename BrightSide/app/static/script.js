// alert("Hello World!"); 

(function(){
    // alert("sagasfgfsdg");
  const http = new XMLHttpRequest();
//   const camera = document.createElement('camera');
  const camera = document.getElementById('camera');

  camera.videoWidth = 200;
  camera.width = 200;
  camera.height = 200;
  camera.videoHeight = 200;
  const enableButton = document.querySelector(".Enable");

  const canvas = document.querySelector('.TEST');
  canvas.style.display = "none";

  const emotionHistoryMemory = 5;
  let counter = 0;
  const frequencies = new Array(20).fill(0);

  //   const canvas = document.querySelector('.TEST');
//   canvas.width = 200;
//   canvas.height = 200;
  const context = canvas.getContext('2d');

  let isEnabled = false;

  function cameraIsOn(){
    if(camera.srcObject != null){
        if(camera.srcObject.getVideoTracks().length > 0){
            return true;
        }
        return false;
    }

    navigator.permissions.query({ name: 'camera' }).then(permissionStatus => {
        if (permissionStatus.state === 'granted') {
          return true;
        }
    });


    return false;
  }

  function disableCamera(){
    // if(camera.srcObject == null){
    //     return;
    // }
    window.alert("disabling camera");

    const tracks = camera.srcObject.getTracks();
    
    tracks.forEach(track => {
        track.stop();
    });

    camera.srcObject = null;
  }

  function toggleCamera(){
    interpretEmotion(0);
    
    if(cameraIsOn()){
        isEnabled = false;
        window.alert("disabling camera");
        // window.alert(isEnabled);
        disableCamera();

        return;
    }

    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        // camera = document.getElementById('camera');
        window.alert("enabling camera");
        camera.srcObject = stream;
        context.drawImage(camera, 0, 0);
        isEnabled = true;
        camera.play();
        // window.alert(isEnabled);
        return;

    }).catch((error) => {
        if (error.name === 'NotAllowedError') {
          window.alert("camera refused");
        //   window.alert(isEnabled);
        }
    });
    
    // isEnabled = !isEnabled;
  }

  /*
    0 is neutral
    1 is happy
    2 is sad
    3 is angry
  */
  async function interpretEmotion(emotion){
    switch(emotion){
        case 0:
            window.alert("switching pages")

            let url = "/meme";
            const response = await fetch(url);
            // http.open("GET", url);
            // http.open(url);
            http.send();
            http.onload = function(){

            }

            break;
        case 1:
            break;
        case 2:
            break;
        case 3:
            break;
        default:
            window.alert("error occurred, emotion in interpretEmotion is not a valid value")
    }
  }

  function capturePhoto(){
    if(cameraIsOn()){
        // window.alert("enabled, capturing photo");  
        // interpretEmotion(0);
        // const image = document.querySelector(".center");

        window.alert(camera.srcObject == undefined || camera.srcObject == null);  
        canvas.width = camera.videoWidth;
        canvas.height = camera.videoHeight;

        context.drawImage(camera, 0, 0);
        // context.drawImage(image, 0, 0);
        window.alert("enabled, capturing photo");  

        // const imageData = canvas.toDataURL('image/png');
        const imageData = canvas.toDataURL("image/png");

        let url = "/predict"
    
        http.open("POST", url) 
      
        http.send(imageData)

        http.onload = function(){
          
            if (http.status === 200){
                const response = http.responseText
                window.alert(response);  
                frequencies[response]++;
                counter++;
                if(counter >= emotionHistoryMemory){
                    counter = 0;
                    let mostFrequent = 0;
                    frequencies.forEach(x => mostFrequent = Math.max(mostFrequent, x));
                    interpretEmotion(mostFrequent);
                }
              // predictionElement.textContent = `predicted class from the model: ${path}`; 
            }else{
                window.alert("error occured");  
            }
             
          }
        
        }
    }  
    


  

  enableButton.addEventListener("click", toggleCamera);
  var intervalID = window.setInterval(capturePhoto, 10000);
})();