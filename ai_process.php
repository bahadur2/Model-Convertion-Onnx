<?php

header("Expires: Tue, 01 Jan 2000 00:00:00 GMT");
header("Last-Modified: " . gmdate("D, d M Y H:i:s") . " GMT");
header("Cache-Control: no-store, no-cache, must-revalidate, max-age=0");
header("Cache-Control: post-check=0, pre-check=0", false);
header("Pragma: no-cache");

// get correct file path
$fileName = $_GET['name'];
$filePath = 'uploads/'.$fileName;


$curl = curl_init();

curl_setopt_array($curl, array(
  CURLOPT_URL => 'http://10.2.4.2:6000/Detection_API/api/v1/detect',
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_ENCODING => '',
  CURLOPT_MAXREDIRS => 10,
  CURLOPT_TIMEOUT => 0,
  CURLOPT_FOLLOWLOCATION => true,
  CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
  CURLOPT_CUSTOMREQUEST => 'POST',
  CURLOPT_POSTFIELDS => array('file'=> new CURLFile($filePath)),
));

$response = curl_exec($curl);

curl_close($curl);

$img = imagecreatefromstring($response);
    imagejpeg($img,'tempcover.jpg', 100);
?>

<!--HTML-->

<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body, html {
  height: 100%;
  margin: 0;
}

.bg {
  /* The image used */
  background-image: url("tempcover.jpg");

  /* Full height */
  height: 100%; 

  /* Center and scale the image nicely */
  background-position: center;
  background-repeat: no-repeat;
  background-size: contain;
  background-color: #000000;
}
</style>
</head>
<body>

<div class="bg"></div>

<p>The File will not the saved.</p>

</body>
</html>
