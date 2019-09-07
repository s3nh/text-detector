function readURL(input) {
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
      $('.image-upload-wrap').hide();

      $('.file-upload-image').attr('src', e.target.result);
      $('.file-upload-content').show();

      $('.image-title').html(input.files[0].name);
    };

    reader.readAsDataURL(input.files[0]);

  } else {
    removeUpload();
  }
}

function removeUpload() {
  $('.file-upload-input').replaceWith($('.file-upload-input').clone());
  $('.file-upload-content').hide();
  $('.image-upload-wrap').show();
}
$('.image-upload-wrap').bind('dragover', function () {
		$('.image-upload-wrap').addClass('image-dropping');
	});
	$('.image-upload-wrap').bind('dragleave', function () {
		$('.image-upload-wrap').removeClass('image-dropping');
});


class TypeWriter {
   constructor(element) {
      this.element = element;
      this.text = element.textContent;
      this.height = element.offsetHeight;
      this.index = 0;
      this.addSpeed = 60;
      this.clearSpeed = 20;
      this.waitTime = 1000;
     
      this.removeText = this.removeText.bind(this);
      this.addText = this.addText.bind(this);
     
      this.init();   
   }
  
    removeText() {
      this.element.textContent = this.element.textContent.slice(0, -1);
	
      if (this.element.textContent.length == 0) {
        clearInterval(this.interval);
        
        var that = this;

        setTimeout(function(){
          that.index = 0;
          that.interval = setInterval(that.addText, that.addSpeed);
        }, this.waitTime);
      }
    }
  
    addText() {
      this.element.textContent += this.text[this.index];

      this.index = this.index + 1;

      if (this.index > this.text.length -1) {
        clearInterval(this.interval);
        
        var that = this;

        setTimeout(function(){
          that.interval = setInterval(that.removeText, that.clearSpeed);
        }, this.waitTime);
      }

    }
  
    init() {
      this.element.textContent = '';
      this.element.style.height = '' + this.height + 'px';
      
      this.interval = setInterval(this.addText, this.addSpeed);
    }
}

Array.from(document.getElementsByClassName("typewriter")).forEach(
    function(element) {
        new TypeWriter(element);
    }
);