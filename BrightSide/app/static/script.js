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
    window.alert(isEnabled);    
    // document.write("hi")
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

  function capturePhoto(){
    if(cameraIsOn()){
        // window.alert("enabled, capturing photo");  

        const image = document.querySelector(".center");

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