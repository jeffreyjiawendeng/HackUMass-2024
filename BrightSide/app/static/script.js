// alert("Hello World!"); 

(function(){
    // alert("sagasfgfsdg");
  const http = new XMLHttpRequest();

  const enableButton = document.querySelector(".Enable");
  let isEnabled = false;

  function toggleEnabled(){
    window.alert(isEnabled);    
    // document.write("hi")
    isEnabled = !isEnabled;
  }

  console.log("asgjkshkgskfg");
  enableButton.addEventListener("click", toggleEnabled);

})();