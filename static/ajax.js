$(document).on('submit','#encode',function(e)
{
    e.preventDefault();
    $.ajax({
        type:'POST',
        url:'/',
        data:{
            offline_layers:$("#offline_layers").val(),
            input_text:$("#input_text").val()
        },
        success:function(encoded_text)
        {
           $("#encoded_text").html(
               "<h2>Torch out (ground truth):</h2><p class=\"form-text\">" + encoded_text + "</p>"
           );
        
           $("#inject-js").html(
               "Loading GPT-2's splitcompute output...<br>"
           );

           gpt2_runner(parseInt($("#offline_layers").val()));
        }
    })
});
