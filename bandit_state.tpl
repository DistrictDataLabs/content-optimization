
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Lato test</title>
    <link href="latofonts.css" rel="stylesheet" type="text/css">
    <link href="latostyle.css" rel="stylesheet" type="text/css">
    <style type="text/css"></style>
</head>

<h1>{{message}}</h1>
<h2>Current state of the bandit arms</h2>
<p>
	
	% for arm in arm_data:
	<div>
		{{arm[0]}}, {{arm[1]}}
	</div>
	% end
</p>