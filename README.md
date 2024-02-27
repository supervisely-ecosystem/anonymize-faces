<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/119248312/ecec1b80-bbb9-4b5b-8d61-05f11723b69d"/>

# Face detection and anonymization
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Results">Results</a> 
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/anonymize-faces)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/anonymize-faces)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/anonymize-faces.png)](https://supervisely.com)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/anonymize-faces.png)](https://supervisely.com)

</div>

## Overview

This application detects faces on images and anonymizes them. It is useful when you need to hide faces on images for privacy reasons.
You can choose between two anonymization methods: solid and blur. You can also choose the shape of the anonymization area: rectangle or ellipse.
A new project will be created with anonymized images. The original images will not be changed.

In some cases you may want to review the results and add/edit annotations. For example, if the app didn't detect some faces, you can add them manually. Or if the app detected a face where there is none, you can delete the annotation. You can also edit the anonymization area if needed. You can do this by following these steps:
1. Run the app with "Save Detections" option enabled.
2. Review the results add/edit annotations if needed.
3. Run the app with "Anonymize" option enabled.

### Release 1.1.0 update

- Renamed app to "Face detection and anonymization"
- Added "Threshold" option
- Added "Save Detections" option for images.
  - This option allows to save detected faces as annotations
  - If Anonymize checkbox is enabled, not only detected faces will be obscured, but all bounding boxes of class `face` will be obscured as well

## How To Run

### Option 1. From ecosystem

0. Find this app in Ecosystem and click `Run Application` button

1. Select Project or Dataset with images you want to anonymize

2. Select the anonymization method: solid or blur

3. Select the shape of the anonymization area: rectangle or ellipse

4. Click `Run` button

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/119248312/770000ce-675c-436b-a8ed-2fb34b8ce63d" width="600"/>

### Option 2. From context menu

0. Go to the list of projects in your Workspace or the list of datasets in a project

1. Open the context menu of a project or dataset, select the application in `Run App -> Transform` section

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/119248312/b6e96a47-a9b0-4ace-82e9-7bab563d5756"/>

2. Select the anonymization method: solid or blur

3. Select the shape of the anonymization area: rectangle or ellipse

4. Click `Run` button

## Results

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/119248312/08481f94-2cfb-4ba4-85d8-10b17cf467d1"/>

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/119248312/55c3f057-1fb7-482a-844b-d742caa09a4e"/>

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/61844772/3f9b0900-206a-43bf-be38-57b80c61c49d"/>