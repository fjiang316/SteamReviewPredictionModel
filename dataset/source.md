# Source of Data
The dataset used for this project is downloaded from reviews on Steam. The specific section we used are provided by staffs of CSE 258 from UCSD, and can be downloaded [here](https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/assignment1.tar.gz).

## Data Description
The dataset contains a total of 175,000 instances of reviews to be used for training. This data is used for both predicting whether a user will play a game or not (task 1), and how long the interaction might last (task 2).

## Structure
| Feature | Description|
|---------|------------|
|**userID** | The ID of the user. This is a hashed user identifier from Steam.|
|**gameID** | The ID of the game. This is a hashed game identifier from Steam.|
|**text** | Text of the user’s review of the game.|
|**date** | Date when the review was entered.|
|**hours** | How many hours the user played the game.|
|**hours_transformed** | log2(hours+1). This transformed value is the one we are trying to predict.|
