// alert("Hello World!"); 

(function(){
    // alert("sagasfgfsdg");
  const http = new XMLHttpRequest();
//   const camera = document.createElement('camera');
  const camera = document.getElementById('camera');

  camera.videoWidth = 200;
  camera.videoHeight = 200;
  const enableButton = document.querySelector(".Enable");

  const canvas = document.querySelector('.TEST');
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

    window.alert("camera is not on from cameraIsOn()");    

    return false;
  }

  function disableCamera(){
    // if(camera.srcObject == null){
    //     return;
    // }

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
        camera.srcObject = stream;

        camera.play();
        isEnabled = true;
        window.alert("enabling camera");
        context.drawImage(camera, 0, 0, canvas.width, canvas.height);
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
        window.alert("enabled, capturing photo");  

        // canvas.width = 200;
        // canvas.height = 200;
        context.drawImage(camera, 0, 0);

        // const imageData = canvas.toDataURL('image/png');
        const imageData = canvas.toDataURL("image/png");
        // const image = document.createElement('img');
        // image.src = imageData;
        const image = document.querySelector(".center");
        // image.src = imageData;
        // document.body.appendChild(image);

        
        const a = document.createElement('a');
        a.href = imageData;
        a.download = "test.png";
        a.click();
        window.alert("end capture photo");  
        localStorage.setItem(".center", image);
    }  


  }

  enableButton.addEventListener("click", toggleCamera);
  var intervalID = window.setInterval(capturePhoto, 10000);
})();