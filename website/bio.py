info = {
   "Pronoun": "he/him",
   "Name": "Amr",
   "Sidebar Name": "<div id=\"sidebar_name\">Amr Farghaly</div>",
   "Sidebar About": "<div id=\"sidebar_about\">AI Engineer and master’s graduate from Saarland University.</div>",
   "About": "Master’s Visual Computing graduate from Saarland University.",
   "Photo": "https://media.licdn.com/dms/image/D5603AQFSOuzB-D346w/profile-displayphoto-shrink_800_800/0/1710946375011?e=1717027200&v=beta&t=ryMO3QVT5OoCkfu0tQaOECiUzNkbPyBfT7VL4KnQvkQ",
   "Email": "amribrahim.amer@gmail.com"
}

esg_project = {
    "Title": "ESG Certification Recommendation System",
    "demo_link": "https://esg-certification-recommender-system.streamlit.app/",
    "Description": """This project aimed to improve upon the manual process of certifying IT products for Ecolabels (Energy Star, TCO) by leveraging Large Language Models (LLMs). LLMs are gaining popularity for the ability to process and analyze large amounts of data. Given product and certification data, these models use Natural Language Processing (NLP) methods to assess whether a product meets the criteria for an ESG certification.""",
    "image": "./assets/ESG.png",
    "image_caption": "High level architecture of the ESG Certification Recommendation System",
    "video1_link": "https://youtu.be/hYOZidi2hwY",
    "video1_desc": """The first step was to create column definitions for the product dataset. Column definitions allowed us to take a certification mandate and rank product attributes by semantic meaning, thereby providing the LLMs with only relevant product information when assessing compliance.""",
    "video2_link": "https://youtu.be/FNa1zimxNJk",
    "video2_desc": """The next step was to query the LLMs for an assessment. The application runs through all mandates (9 for Energy Star, 17 for TCO), and passes the mandate description along with the five most relevant product features to the LLMs, which output an assessment and reasoning. The percentage of mandates passed serves as the final output or confidence that a product is eligible for certification. This, along with the LLM reasoning, is output to the user for review. """
}

fifa_project = {
    "Title": "ANALYZE & PREDICT 2022 FIFA WORLD CUP",
    "Description": """The goal of this project wass to utilize machine learning classification algorithms to create an interface showcasing national team performance and predict the outcome of matchups in the 2022 FIFA World Cup tournament. We leveraged data scraping to pull match data from SoccerBase and player ratings from FIFA to train and test five ML classifiers on predicting match results given recent team form statistics and player ratings. We found that the best performing model was the AdaBoost Classifier, which had a test accuracy of 67.44%.""",
    "image": "./assets/fifa_project.png", 
    "image_caption": "Final tournament prediction using AdaBoost, the best performing classification algorithm"
}

meta_project = {
    "Title": "The Effect of Facebook Ad Spending on Elections: Research on the 2020 United States Presidential Elections",
    "Description": """Ad spending on social media is projected to reach over $173 billion in 2022, accounting for 33% of all digital advertising spend. This project aimed to examine political advertisement spending on Facebook and the impact it had on the 2020 election. Based on our analysis and experiments, we found that ad spending was associated with a larger voter turnout. Further, we found that ads for the Trump campaign seemed more effective than Biden ads at bringing out the vote for the 2020 election, where a 1% increase in Trump's ad spend translated to 584 votes won. The only region where Biden's ads seemed effective in increasing voter turnout was for the west coast, where Biden gained 1007 votes for every 1% increase in ad spend.""",
    "image": "./assets/meta_project.png", 
    "image_caption": "Undecided Voters Won ~ log(Candidate 2020 Ad Spend) for Trump and Biden"
}