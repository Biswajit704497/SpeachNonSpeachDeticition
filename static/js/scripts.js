  document.onkeydown = function(e) {
    if (e.keyCode == 123) { // F12
      return false;
    } else if (e.ctrlKey && e.shiftKey && e.keyCode == 73) { // Ctrl+Shift+I
      return false;
    } else if (e.ctrlKey && e.keyCode == 85) { // Ctrl+U
      return false;
    }
  };

