import Container from "@/components/Container";
import { useEffect, useRef, Suspense, useState } from "react";
import styles from "@/styles/Home.module.css";
import { Button } from "@/components/ui/button";
import {
  ChevronRight,
  Code2,
  Frame,
  SearchCheck,
  Eye,
  MonitorSmartphone,
} from "lucide-react";
import { TriangleDownIcon } from "@radix-ui/react-icons";
import Spline from "@splinetool/react-spline";
import Link from "next/link";
import { cn, scrollTo } from "@/lib/utils";
import Image from "next/image";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
  type CarouselApi,
} from "@/components/ui/carousel";
import VanillaTilt from "vanilla-tilt";
import { motion } from "framer-motion";
import {
  BarChart3,
  Sparkles,
  Camera,
  Sigma,
  Server,
} from "lucide-react";


const aboutStats = [{ label: "Years of experience", value: "2+" }, { label: "Technologies & frameworks used", value: "20+" }, { label: "Companies worked with", value: "3+" }, { label: "End-to-end projects delivered", value: "10+" }];

const projectDetails = [
  {
    title: "Nutrigence Bot - AI-Powered Nutrition Assistant",
    description: "Advanced conversational AI system for personalized child and adolescent nutrition guidance",
    fullDescription: "Developed an intelligent nutrition chatbot using RAG (Retrieval-Augmented Generation) architecture with Mistral-7B-Instruct LLM, capable of providing evidence-based dietary recommendations, personalized meal planning, and nutritional guidance based on USDA dietary guidelines with high accuracy. Architecture Highlights: Model: Mistral-7B-Instruct (4-bit quantized),Embedding Model: BAAI/bge-small-en, Vector Store: FAISS with 10 similar document retrieval",
    technologies: ["Python", "Streamlit", "LangChain", "FAISS", "Llama.cpp", "Pandas","HuggingFace"],
    features: [
      "Context-aware nutritional guidance using RAG architecture",
      "Structured dietary recommendations by age, sex, and calorie level",
      "Interactive query system with predefined and custom questions",
      "Multi-category nutrition data:Daily food group targets, Weekly protein subgroup recommendations, Vegetable subgroup intake guidelines, Discretionary calorie limits ",
      "Demographic-based personalization (age, sex, activity level)",
      "Vector similarity search for relevant context retrieval"
    ],
    metrics: [
      { label: "Response Accuracy (based on USDA guidelines)", value: "90%+" },
      { label: "Avg Response Time", value: "<3 seconds" },
      { label: "Data Coverage (kcal range across multiple age groups) ", value: " 1,600-3,200 kcal" }
    ],
    github: "https://github.com/ftkiranraj08/Nutrigence-Bot",
    live: "#",
    category: "RAG/AI"
  },
  {
    title: "Real-Time Tweet Sentiment Pipeline",
    description: "End-to-end streaming sentiment analysis system for social media using Spark, Delta Lake, and Transformers",
    fullDescription: "Built a production-scale real-time sentiment analysis pipeline for Twitter data using Spark Structured Streaming and a medallion architecture in Delta Lake. Integrated a Hugging Face transformer model via MLflow to classify tweet sentiment with high accuracy, enabling live dashboards and aggregated insights.",
    technologies: ["PySpark", "Delta Lake", "Hugging Face", "MLflow", "Python", "Databricks"],
    features: [
      "Spark Streaming with Delta Lake medallion architecture",
      "Real-time sentiment scoring using transformer models",
      "Automated data cleaning and mention extraction",
      "MLflow-integrated model serving and UDF deployment",
      "Live monitoring with throughput and latency tracking"
    ],
    metrics: [
      { label: "Pipeline Latency", value: "< 2 seconds" },
      { label: "Model Inference Throughput", value: "1K+ tweets/sec" },
      { label: "Daily Processed Tweets", value: "5M+" }
    ],
    github: "https://github.com/ftkiranraj08/Data-Science-at-Scale-Final-Project",
    live: "#",
    category: "STREAMTWEET"
  },
  {
    title: "Hierarchical Bayesian Media Mix Modeling",
    description: "Geo-aware marketing optimization system for regional advertising effectiveness",
    fullDescription: "Developed a hierarchical Bayesian Media Mix Model to quantify the causal impact of advertising spend across 169 Designated Market Areas (DMAs). The model integrates nonlinear media transformations (adstock, saturation) with contextual clustering and partial pooling, enabling stable, region-specific ROI estimates and data-driven budget allocation.",
    technologies: ["PyMC", "JAX", "Python", "Pandas", "Scikit-learn", "LightweightMMM","Matplotlib","arviz"],
    features: [
      "Hierarchical Bayesian modeling with partial pooling across 169 DMAs",
      "Nonlinear media response modeling (adstock & saturation effects)",
      "Contextual clustering using demographic and socioeconomic features",
      "Full probabilistic uncertainty quantification for all estimatess",
      "Data-driven budget allocation recommendations",
      "Scalable inference using PyMC with JAX acceleration"
    ],
    metrics: [
      { label: "Out-of-Sample R²", value: "0.77" },
      { label: "DMA-Level ROI Precision", value: "+35% vs. baseline" },
      { label: "Markets Modeled", value: "169 DMAs" }
    ],
    github: "#",
    live: "#",
    category: "HIERARCHICAL MMM"
  },
  {
    title: "Multilingual Translator with Sentiment & Summarization",
    description: "Real-time speech-to-text translator with emotional tone analysis and text summarization",
    fullDescription: "Developed a multi-functional NLP pipeline that translates spoken or written text between languages (Hindi, English, Tamil) in real-time, performs sentiment analysis to detect emotional tone, and summarizes long documents using the RoBERTa model. The system integrates speech recognition, Google Translate API, and transformer models for end-to-end multilingual communication.",
    technologies: ["Python", "RoBERTa", "Transformers", "NLTK", "SpeechRecognition", "Google Translate API", "PyTorch"],
    features: [
      "Real-time speech-to-text translation with sentiment analysis (positive/negative/neutral)",
      "Text summarization using RoBERTa model for English documents",
      "Multilingual support for English, Hindi, and Tamil",
      "Integration of Google Translate API for improved contextual accuracy",
      "End-to-end NLP pipeline from audio input to summarized/translated output"
    ],
    metrics: [
      { label: "Translation Speed", value: "<300 ms" },
      { label: "Supported Languages", value: " 3+" },
      { label: "Summarization Accuracy", value: "Contextually coherent" }
    ],
    github: "https://github.com/yourusername/rag-assistant",
    live: "#",
    category: "NLP"
  },
  {
    title: "Playlist Optimization & Recommendation Analysis",
    description: "Statistical & non-parametric exploration of track popularity, genre diversity, and playlist structure.",
    fullDescription: "Conducted an in-depth analysis of the Spotify Million Playlist Dataset (MPD) to uncover the key factors influencing playlist engagement and user preferences. Applied statistical tests, clustering, and visualization techniques to identify patterns in genre distribution, track repetition, and sequence logic, providing actionable insights to enhance music recommendation systems.",
    technologies: ["Python", "Pandas", "NumPy", "SciPy", "Scikit-learn", "Matplotlib", "Seaborn", "Plotly"],
    features: [
      "Statistical and non-parametric analysis of 1M+ Spotify playlists",
      "Genre diversity and track popularity trend identification",
      "Clustering and collaborative filtering for recommendation insights",
      "Interactive visualizations of playlist structure and user behavior",
      "Data-driven strategies to improve playlist curation and engagement"
    ],
    metrics: [
      { label: "Playlists Analyzed", value: " 1,000,000+" },
      { label: "Genres Identified", value: "2,000+" },
      { label: "Recommendation Model Precision", value: "Improved by data insights" }
    ],
    github: "https://github.com/ftkiranraj08/Playlist-Recommender",
    live: "#",
    category: "SPOTIFY MPD"
  },
  {
    title: "Spatiotemporal Network Analysis of Urban Mobility",
    description: "Modeling NYC taxi & ride-hailing dynamics using graph theory and econometrics",
    fullDescription: "Applied network science to model and analyze NYC's taxi and for-hire vehicle (FHV) networks using a decade of trip data. Constructed spatiotemporal graphs, computed centrality metrics (PageRank, Betweenness), and performed panel regressions to uncover relationships between structural zone importance and fare economics, offering data-driven insights for congestion pricing and fleet optimization.",
    technologies: ["Python", "Pandas", "NetworkX", "Statsmodels", "Dash", "GeoPandas", "SciPy", "Plotly"],
    features: [
      "Spatiotemporal network modeling of 3+ billion NYC taxi/FHV trips",
      "Centrality analysis using PageRank, Betweenness, and degree measures",
      "Panel regression linking network structure to fare economics",
      "Interactive dashboard for real-time network visualization",
      "Policy insights for congestion pricing and equitable fleet allocation"
    ],
    metrics: [
      { label: "Trips Analyzed", value: " 3 Billion+" },
      { label: "Zones Modeled", value: "263+" },
      { label: "Model R² Up to", value: "0.24" }
    ],
    github: "https://github.com/ftkiranraj08/Analyzing-NYC-Taxi-and-Ride-Hailing-Dynamics/tree/main",
    live: "#",
    category: "SPOTIFY MPD"
  },
  {
    title: "Data-Driven Market Potential & Customer Engagement Strategy",
    description: "Advanced data mining and clustering of U.S. supermarket data to uncover sales and profit potential",
    fullDescription: "Conducted a comprehensive data mining analysis on merged supermarket and population datasets to identify high-potential U.S. states for business expansion. Applied clustering, classification, and information gain techniques to evaluate sales per capita, profit per capita, and product diversity, providing actionable insights for targeted marketing and investment decisions.",
    technologies: ["Python", "Pandas", "Scikit-learn", "PCA", "K-Means", "Matplotlib", "Seaborn"],
    features: [
      "Multi-metric state ranking based on sales and profit per capita",
      "K-Means clustering to identify high-potential market segments",
      "Information gain analysis to determine key profitability drivers",
      "Classification framework for high/moderate/low potential states",
      "Diversity analysis to recommend broad vs. niche retail strategies"
    ],
    metrics: [
      { label: "States Classified", value: " 48" },
      { label: "Profit Correlation Score", value: "0.83" },
      { label: "High-Potential Markets Identified", value: "6" }
    ],
    github: "https://github.com/ftkiranraj08/Mapping-market-dynamics",
    live: "#",
    category: "RETAIL ANALYTICS"
  },
  {
    title: "COVID-19 Case Prediction with Social Media Awareness Signals",
    description: "Machine learning framework integrating Twitter trends to forecast county-level infections in Ohio",
    fullDescription: "Developed an ensemble ML pipeline to predict COVID-19 cases across Ohio counties by combining epidemiological data with Twitter-derived social awareness metrics. Engineered features from normalized similarity scores (Jaccard, Cosine, Intersection) and applied stacking of XGBoost, Random Forest, LightGBM, and CatBoost models, achieving high predictive accuracy and offering actionable insights for public health communication.",
    technologies: ["Python", "XGBoost", "Random Forest", "LightGBM", "CatBoost", "Scikit-learn", "Pandas","t-SNE"],
    features: [
      "Integration of Twitter awareness metrics with traditional epidemiological data",
      "Advanced feature engineering including cyclical encoding and interaction terms",
      "Stacking ensemble model combining four high-performance regressors",
      "Dimensionality reduction and clustering for demographic & awareness features",
      "High R² score demonstrating strong predictive power for public health planning"
    ],
    metrics: [
      { label: "Prediction Accuracy R² ", value: " 0.8636" },
      { label: "Counties Analyzed", value: "Ohio (Statewide)" },
      { label: "Features Engineered", value: "144+" }
    ],
    github: "https://github.com/ftkiranraj08/Mapping-market-dynamics",
    live: "#",
    category: "RETAIL ANALYTICS"
  },
  {
    title: "Airline Ticket Fare Forecasting with Temporal & Demand Signals",
    description: "Machine learning framework for predicting flight ticket prices using historical, temporal, and route-level features",
    fullDescription: "Developed an end-to-end machine learning pipeline to forecast airline ticket fares by integrating historical pricing data with temporal, route, and demand-related features. Engineered advanced features including cyclical time encodings, lag variables, rolling statistics, and interaction terms. Applied ensemble learning using Gradient Boosting–based and tree-based regressors to capture complex non-linear pricing patterns, achieving strong predictive performance and practical insights for travel planning and pricing strategy.",
    technologies: ["Python", "XGBoost", "Random Forest", "LightGBM", "CatBoost", "Scikit-learn", "Pandas","matplotlib","numpy"],
    features: [
      "Airline fare prediction using historical, temporal, and route-specific signals",
      "Advanced feature engineering with lag features, rolling averages, and cyclical encoding",
      "Stacking ensemble model combining multiple high-performance regressors",
      "Time-aware train–validation strategy to prevent data leakage",
      "Robust regression performance suitable for pricing intelligence and demand forecasting"
    ],
    metrics: [
      { label: "Prediction Accuracy R² ", value: " 0.8476" },
      { label: "Routes Analyzed", value: "Multi-city / Domestic Flights" },
      { label: "Features Engineered", value: "120+" }
    ],
    github: "https://github.com/ftkiranraj08/Airline-Ticket-Fare-Forecasting/tree/main",
    live: "#",
    category: "RETAIL ANALYTICS"
  },
];

const projects = [
  {
    title: "Nutrigence Bot",
    description: "Nutrigence Bot - AI-Powered Nutrition Assistant",
    image: "/assets/chatbot.webm",
    href: "https://github.com/ftkiranraj08/Nutrigence-Bot",
  },
  {
    title: "Real-Time Tweet Sentiment Pipeline",
    description: "Real-Time Tweet Sentiment Pipeline",
    image: "/assets/My-Movie.webm",
    href: "#",
  },
  // {
  //   title: "TranslateBot",
  //   description: "Powerful Multilingual Translation Bot for Discord",
  //   image: "/assets/translate_bot.webm",
  //   href: "#",
  // },
  {
    title: "Marketing Analytics ",
    description: "Hierarchical Bayesian Media Mix Modeling for Marketing Optimization",
    image: "/assets/Screen-Recording-2026-01-04-at-1 (1).webm",
    href: "#",
  },
  {
    title: "MAGic",
    description: "MAthematical Gene Circuit (MAGiC) Modeling Kits",
    image: "/assets/Walkthrough.webm",
    href: "#",
  },
];

const services = [
  {
    service: "AI & Machine Learning",
    description:
      "Designing and deploying end-to-end machine learning and deep learning systems, including NLP, computer vision, and predictive modeling for real-world applications.",
    icon: Code2,
  },
  {
    service: "Generative AI & RAG Systems",
    description:
      "Building Retrieval-Augmented Generation (RAG) pipelines using LangChain, FAISS, and large language models to enable accurate, explainable, and scalable AI applications.",
    icon: Sparkles,
    
  },
  {
    service: "Data Science & Analytics",
    description:
      "Transforming complex datasets into actionable insights using statistical modeling, clustering, experimentation, and business-driven analytics.",
    icon: Frame,
  },
  {
    service: "Data Engineering & MLOps",
    description:
      "Designing websites that look and perform equally well on all devices and screen sizes.",
    icon: Server,
  },
  {
    service: "Bayesian & Statistical Modeling",
    description:
      "Applying Bayesian hierarchical models, time-series forecasting, and causal inference techniques to quantify uncertainty and optimize strategic decisions.",
    icon: Sigma,
  },

];

// Teaching Experience Data
const teachingExperience = [
  {
    type: "Teaching Assistantship",
    position: "Graduate Teaching Assistant",
    institution: "University of Rochester - Simon Business School",
    duration: "Jan 2025 - May 2025",
    location: "Rochester, New York",
    courses: [
      {
        code: "CIS431",
        name: "Big Data",
        duration: "Mar 2025 - May 2025 · 3 mos",
        topics: [
          "Big Data Fundamentals: Introduction to Big Data concepts and architecture overview",
          "HDFS & MapReduce: Distributed storage with HDFS and the MapReduce programming model",
          "Apache Hive: Querying data with HiveQL, data management in Hive, relational analysis & complex data types, and text-analysis workflows",
          "Apache Spark: Core Spark architecture, data inspection, transformations & actions, and combining & grouping large datasets",
          "Spark MLlib: Machine-learning fundamentals in Spark MLlib, feature extraction & building regression models, classification models & hyperparameter tuning, and cluster models & end-to-end ML pipelines",
          "Streaming Data Analysis: Introduction to real-time data processing with Spark Streaming"
        ]
      },
      {
        code: "GBA468",
        name: "Prescriptive Analytics with Python",
        duration: "Jan 2025 - Mar 2025 · 3 mos",
        topics: [
          "Optimization Models: Linear Programming (LP), Integer LP, Non-Linear Programming (NLP), Multi-Objective Optimization (MOO), and Network Optimization",
          "Decision Analysis: Decision rules under risk & uncertainty, Expected Utility Theory, Decision Trees, and Single/Multistage Decision Models",
          "Simulation Modeling: Monte Carlo Simulation, Capacity Management, and Python-based simulation modeling"
        ]
      }
    ],
    responsibilities: [
      "Assisted instructors with course delivery for Big Data and Prescriptive Analytics courses serving 100+ graduate students",
      "Conducted office hours and provided one-on-one support for students on complex topics including Spark MLlib, optimization models, and simulation",
      "Graded assignments, projects, and exams ensuring timely feedback and maintaining academic standards",
      "Facilitated hands-on lab sessions covering Apache Spark, Hive, MapReduce, and Python-based optimization frameworks",
      "Helped students debug code, troubleshoot distributed computing environments, and understand machine learning pipelines",
      "Developed supplementary materials and examples to clarify challenging concepts in streaming data analysis and decision trees"
    ],
    technologies: ["Apache Spark", "Hadoop", "HDFS", "MapReduce", "Apache Hive", "Spark MLlib", "Python", "PySpark", "Optimization Algorithms", "Monte Carlo Simulation"],
    impact: "Supported graduate student learning in advanced data engineering and prescriptive analytics, contributing to strong course evaluations"
  }
];

// Research Experience Data
const researchExperience = [
  {
    position: "Research Assistant",
    institution: "University of Rochester Medical Center / Dr. Dong Mei Li's Lab",
    duration: "Aug 2024 - Dec 2025",
    location: "Rochester, NY",
    description: "Conducted interdisciplinary public health research combining statistical modeling, machine learning, and multimodal social media analysis to study the effects of smoking, vaping, and comorbidities on COVID-19 severity and tobacco use behaviors.",
    responsibilities: [
      "Conducted advanced statistical analysis on de-identified N3C Level 2 clinical data to examine moderation effects of smoking and vaping on the relationship between comorbidities and COVID-19 outcomes including hospitalization, ICU admission, and mortality",
      "Developed and implemented Structural Equation Modeling (SEM) and Stochastic Latent Effect Modeling (SLEM) in MPlus to analyze mediation and causal pathways linking smoking, comorbidities, and COVID-19 severity",
      "Performed cross-tabulations and demographic stratified analyses to study interactions between smoking status, HIV status, comorbidities, and COVID-19 outcomes, identifying key health disparity patterns",
      "Built statistical models and generated publication-quality visualizations to support findings on inequities in smoking/vaping behaviors and COVID-19 severity across demographic groups",
      "Leveraged TikTok Research API to collect and preprocess large-scale vaping-related video data, extracting speech and on-screen text using OpenAI Whisper for speech-to-text and EasyOCR for text recognition",
      "Implemented Video LLaMA for multimodal classification of TikTok content into pro-vaping and anti-vaping categories, contributing behavioral insights for public health communication strategies",
      "Analyzed PATH Wave 7 Adult Survey data (30,000+ respondents, 1,900+ variables) to model tobacco use patterns across non-users, single-product users, and multi-product users",
      "Engineered a comprehensive feature set across 10 behavioral domains and performed systematic feature selection, multicollinearity reduction (VIF < 5), and dimensionality control to improve model interpretability",
      "Addressed severe class imbalance using SMOTE and class-weighted learning, developing ensemble classifiers combining XGBoost, LightGBM, and CatBoost with soft voting",
      "Conducted rigorous evaluation using stratified cross-validation, AUC/AUPRC analysis, confusion matrices, and benchmarking against prior models to validate robustness and generalizability",
      "Collaborated with multidisciplinary research teams and presented findings in lab meetings and research seminars"
                    ],
    technologies: [
      "Python", "Pandas", "NumPy", "Scikit-learn", "XGBoost", "LightGBM", "CatBoost",
      "imbalanced-learn", "SciPy", "MPlus", "Matplotlib", "OpenAI Whisper",
      "EasyOCR", "Video LLaMA"
                ],
    publications: [
      "Manuscript under preparation on smoking/vaping moderation effects in COVID-19 severity (Target: Public Health / Epidemiology Journal)"
      ],
    impact: "Produced data-driven insights on tobacco use, COVID-19 severity, and health disparities."
  },
  {
    position: "Graduate Research Assistant / Software Developer",
    institution: "University of Rochester / Dr. Allison J Lopatkin's Lab",
    duration: "Jan 2025 - Dec 2025",
    location: "Rochester, NY",
    description: "Developed a full-stack web application for designing and simulating genetic circuits with hardware integration for physical EEPROM-based component storage and automated DNA circuit modeling.",
    responsibilities: [
      "Built Flask-based backend with ODE solver for genetic circuit simulation using Hill kinetics",
      "Implemented drag-and-drop interface with jsPlumb for visual circuit design (1000+ lines frontend code)",
      "Developed bidirectional hardware integration for EEPROM multiplexer boards (32 channels, MUX A/B)",
      "Created automated circuit ontology builder parsing regulatory relationships and component interactions",
      "Engineered real-time parameter tuning system with dial controls for promoter/RBS strength adjustment",
      "Implemented circuit-to-hardware mapping system converting visual designs to physical board layouts",
      "Built comprehensive logging and debugging system for EEPROM read/write operations",
      "Developed LaTeX equation rendering for dynamical system visualization"
    ],
    technologies: [ "Python",
    "Flask",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "JavaScript",
    "jQuery",
    "jsPlumb",
    "Bootstrap 5",
    "HTML5/CSS3",
    "Serial Communication",
    "ODE Solvers" ],
    publications: ["MAGiC: A Modular Genetic Circuit Design Platform (AIChE Annual Student Conference, 2025)"],
    impact: "Submitted for patent protection through UR Ventures."
  }
];

// Internships Data
const internships = [
  {
    role: "Data Science Intern",
    company: "Butler & Till",
    duration: "Aug 2025 - Dec 2025",
    location: "Rochester, United States",
    type: "Internship, Capstone Project",
    description: "Developed and deployed ML models for production recommendation systems serving millions of users.",
    achievements: [
      "Engineered a Geo-level Bayesian Hierarchical Media Mix Model (GBHMMM) using PyMC across 188 U.S. DMAs to quantify regional media effectiveness and optimize local budget allocation",
      "Built a DMA × Week panel dataset (~18K rows) integrating per-capita sales, demographics, and media spend across TV, Video, Social, Display, and OOH channels",
      "Designed and implemented Adstock (carryover) and Hill (saturation) transformations to model diminishing returns and lagged media effects",
      "Conducted baseline OLS and Ridge regressions with DMA fixed effects achieving R² = 0.833 and RMSE = 0.009 to validate signal stability",
      "Applied Bayesian hierarchical pooling and prior regularization to reduce cross-brand ROAS variance by ~60–65% and response-curve bias by ~75%, producing significantly narrower credible intervals"
    ],
    technologies: ["PyMC", "JAX", "Python", "Pandas", "Scikit-learn", "LightweightMMM","Matplotlib","arviz"],
    skills: ["Bayesian Modeling", "Media Mix Modeling", "Causal Inference", "Regression Analysis", "Marketing Analytics"],
    impact: "Enabled data-driven media budget optimization through robust, interpretable Bayesian modeling of regional media response"
  },
  {
    role: "Data Science Intern",
    company: "Mendon Group LLC",
    duration: "Jun 2025 - Jul 2025",
    location: "Rochester, United States",
    type: "Internship",
    description: "Built an AI-driven food compliance and recommendation system integrating large-scale product databases, regulatory rules, and multimodal machine learning.",
    achievements: [
    "Engineered an AI-powered Smart Snack Compliance Calculator using ~1.2M GS1 product records, achieving 92% rule-matching accuracy",
    "Integrated GS1 product data, USDA FoodData Central APIs, and user product scans to support compliance explanations and healthier substitutions",
    "Developed a RAG-powered food compliance agent using Mistral-7B-Instruct, LangChain, and FAISS, achieving 94% retrieval accuracy across 11+ regulatory thresholds",
    "Implemented a computer vision module using Faster R-CNN to analyze vending machine images and assess compliance across four regulatory frameworks",
    "Authored comprehensive technical documentation and operational runbooks to standardize model updates and data pipelines"
    ],
    technologies: ["Python", "LangChain", "FAISS", "Mistral-7B","LLMs",
  "PyTorch", "Faster R-CNN", "OpenCV",
  "GS1 APIs", "USDA FoodData Central","PostgreSQL","RAG"],
    skills: ["Retrieval-Augmented Generation", "Computer Vision", "NLP", "Regulatory AI", "Data Integration"],
    impact: "Enabled automated, explainable food compliance checks and recommendations at scale using multimodal AI"
  },
  {
   role: "Data Science Intern",
company: "Eco Logistics Solutions Pvt. Ltd.",
duration: "Jul 2023 - Jul 2024",
location: "Chennai, India",
type: "Internship",
description: "Applied customer analytics and business intelligence to improve marketing performance and operational decision-making.",
achievements: [
  "Executed RFM analysis and K-Means clustering on 10,000+ customer transactions to generate actionable customer personas",
  "Delivered segmentation insights that powered targeted A/B marketing campaigns, increasing customer retention by 15%",
  "Developed and deployed a dynamic Tableau dashboard using SQL to track CAC, conversion rates, and regional sales performance",
  "Streamlined BI workflows, reducing manual reporting effort by 30% and enabling faster optimization of digital sales funnels"
],
technologies: ["Python", "Pandas", "Scikit-learn", "SQL", "Tableau"],
skills: ["Customer Segmentation", "Clustering", "Business Intelligence", "A/B Testing", "Marketing Analytics"],
impact: "Improved customer retention and marketing ROI through data-driven segmentation and real-time performance tracking"

  }
];

// Education Data
const education = [
  {
    degree: "Master of Science in Data Science",
    institution: "University of Rochester",
    duration: "Aug 2024 - Dec 2025",
    location: "Rochester, NY",
    gpa: "3.6/4.0",
    coursework: [
      "Time Series Analysis and Forecasting", "Computational Statistics", "Data Mining", "Statistical Machine Learning", "Data Science at Scaling", "Network Science and Analysis","Practicum"
    ],
    achievements: [
      "Graduate Research Assistant",
      "Teaching Assistant for Big Data & Prescriptive Analytics with Python (2 semesters)",
      "30% tuition scholarship recipient"
    ]
  },
  {
    degree: "Bachelor of Technology in Information Technology",
    institution: "Anna University",
    duration: "Sep 2020 - May 2024",
    location: "Tamil Nadu, India",
    gpa: "8.59/10.0",
    coursework: [
      "Artificial Intelligence and Machine Learning", "Data Structures", "Advanced Data Structures", "Design and Analysis of Algorithms", "Cloud and Big Data Analytics", "Database Management Systems" ],
    achievements: [
      "Distinction with Honors",
      "Conducted advanced data structures coding workshops for junior students.",
      "Two-time National Level Hackathon Winner"
    ],
  }
];

// Achievements & Publications Data
const achievements = [
  {
    category: "Publications",
    items: [
      {
        title: "Language Translator with Sentiment Analysis Using The RoBERTa model and NLP",
        authors: "Kiran Raj Paramasivam, T.G. Madhusoodhan and L. Leena Jenifer",
        venue: "7th International Conference on Intelligent Computing (ICONIC)",
        year: "2024",
        type: "Conference Paper",
        status: "Awaiting Publication",
        // link: "https://arxiv.org/paper-link",
        description: "Proposed an NLP-based multilingual text and speech translation system that also detects sentiment, enabling accurate, real-time cross-language communication.",
      },
      {
        title: "Modeling Gene Circuit Dynamics: An Educational Toolkit Integrating Modular Hardware and Predictive Software",
        authors: " Kiran Raj Paramasivam, Chauner Clausing, Elizabeth Martin, Jordan Blair and Allison Lopatkin,",
        venue: "American Institute of Chemical Engineers (AIChE) Annual Student Conference",
        year: "2025",
        type: "Conference Paper",
        status: "Under Review",
        // link: "https://aiche.confex.com/aiche/2025/ascprogram/papers/viewonly.cgi?password=666830&username=724099",
        description: "Introduced MAGiC, an interactive hardware–software toolkit that enables intuitive design, simulation, and visualization of genetic circuit dynamics for synthetic biology education.",
      },
    ]
  },
  
  {
    category: "Patents & IP",
    items: [
      {
        title: "MAthematical Gene Circuit (MAGiC) Modeling Kits",
        number: "US Patent Application 2-25070",
        year: "2025",
        status: "Pending",
        description: "An interactive hardware–software toolkit that enables intuitive design, simulation, and visualization of genetic circuit dynamics for synthetic biology education."
      }
    ]
  },
];

export default function Home() {
  const refScrollContainer = useRef(null);
  const [isScrolled, setIsScrolled] = useState<boolean>(false);
  const [carouselApi, setCarouselApi] = useState<CarouselApi | null>(null);
  const [current, setCurrent] = useState<number>(0);
  const [count, setCount] = useState<number>(0);

  // handle scroll
  useEffect(() => {
    const sections = document.querySelectorAll("section");
    const navLinks = document.querySelectorAll(".nav-link");

    async function getLocomotive() {
      const Locomotive = (await import("locomotive-scroll")).default;
      new Locomotive({
        el: refScrollContainer.current ?? new HTMLElement(),
        smooth: true,
      });
    }

    function handleScroll() {
      let current = "";
      setIsScrolled(window.scrollY > 0);

      sections.forEach((section) => {
        const sectionTop = section.offsetTop;
        if (window.scrollY >= sectionTop - 250) {
          current = section.getAttribute("id") ?? "";
        }
      });

      navLinks.forEach((li) => {
        li.classList.remove("nav-active");

        if (li.getAttribute("href") === `#${current}`) {
          li.classList.add("nav-active");
          console.log(li.getAttribute("href"));
        }
      });
    }

    void getLocomotive();
    window.addEventListener("scroll", handleScroll);

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  useEffect(() => {
    if (!carouselApi) return;

    setCount(carouselApi.scrollSnapList().length);
    setCurrent(carouselApi.selectedScrollSnap() + 1);

    carouselApi.on("select", () => {
      setCurrent(carouselApi.selectedScrollSnap() + 1);
    });
  }, [carouselApi]);

  // card hover effect
  useEffect(() => {
    const tilt: HTMLElement[] = Array.from(document.querySelectorAll("#tilt"));
    VanillaTilt.init(tilt, {
      speed: 300,
      glare: true,
      "max-glare": 0.1,
      gyroscope: true,
      perspective: 900,
      scale: 0.9,
    });
  }, []);

  return (
    <Container>
      <div ref={refScrollContainer}>
        <Gradient />

        {/* Intro */}
        <section
          id="home"
          data-scroll-section
          className="mt-40 flex w-full flex-col items-center xl:mt-0 xl:min-h-screen xl:flex-row xl:justify-between"
        >
          <div className={styles.intro}>
            <div
              data-scroll
              data-scroll-direction="horizontal"
              data-scroll-speed=".09"
              className="flex flex-row items-center space-x-1.5"
            >
              <span className={styles.pill}>Python</span>
              <span className={styles.pill}>Machine Learning</span>
              <span className={styles.pill}>Artificial intelligence</span>
            </div>
            <div>
              <h1
                data-scroll
                data-scroll-enable-touch-speed
                data-scroll-speed=".06"
                data-scroll-direction="horizontal"
              >
                <span className="text-6xl tracking-tighter text-foreground 2xl:text-8xl">
                  Hello, I&apos;m
                  <br />
                </span>
                <span className="clash-grotesk text-gradient text-6xl 2xl:text-8xl">
                  Kiran Raj Paramasivam.
                </span>
              </h1>
              <p
                data-scroll
                data-scroll-enable-touch-speed
                data-scroll-speed=".06"
                className="mt-1 max-w-lg tracking-tight text-muted-foreground 2xl:text-xl"
              >
                A data scientist and ML engineer building end-to-end AI systems with real-world impact.
              </p>
            </div>
            <span
              data-scroll
              data-scroll-enable-touch-speed
              data-scroll-speed=".06"
              className="flex flex-row items-center space-x-1.5 pt-6"
            >
              <Link href="mailto:ftkiranraj@proton.me" passHref>
                <Button>
                  Get in touch <ChevronRight className="ml-1 h-4 w-4" />
                </Button>
              </Link>
              <Button
                variant="outline"
                onClick={() => scrollTo(document.querySelector("#about"))}
              >
                Learn more
              </Button>
            </span>

            <div
              className={cn(
                styles.scroll,
                isScrolled && styles["scroll--hidden"],
              )}
            >
              Scroll to discover{" "}
              <TriangleDownIcon className="mt-1 animate-bounce" />
            </div>
          </div>
          <div
            data-scroll
            data-scroll-speed="-.01"
            id={styles["canvas-container"]}
            className="mt-14 h-full w-full xl:mt-0"
          >
            <Suspense fallback={<span>Loading...</span>}>
              <Spline scene="/assets/scene.splinecode" />
            </Suspense>
          </div>
        </section>

        {/* About */}
        <section id="about" data-scroll-section>
          <div
            data-scroll
            data-scroll-speed=".4"
            data-scroll-position="top"
            className="my-14 flex max-w-6xl flex-col justify-start space-y-10"
          >
            <h2 className="py-16  pb-2 text-3xl font-light leading-normal tracking-tighter text-foreground text-justify xl:text-[40px]">
              I&apos;m an experienced data scientist specializing in building end-to-end AI and machine learning systems, with hands-on experience since 2023. My work spans startups, consulting, and academic research, where I’ve led projects from problem formulation and data engineering through model development, validation, and deployment. I’ve built large-scale data pipelines, RAG-powered agents, and advanced ML models across NLP, computer vision, and Bayesian modeling, while collaborating closely with cross-functional teams to translate complex data into measurable business impact.

              {/* <Link
                href="https://pytorch.org/"
                target="_blank"
                className="underline"
              >
                  PyTorch & Hugging Face
              </Link>{" "} */}
            </h2>
            <div className="grid grid-cols-2 gap-8 xl:grid-cols-3">
              {aboutStats.map((stat) => (
                <div
                  key={stat.label}
                  className="flex flex-col items-center text-center xl:items-start xl:text-start"
                >
                  <span className="clash-grotesk text-gradient text-4xl font-semibold tracking-tight xl:text-6xl">
                    {stat.value}
                  </span>
                  <span className="tracking-tight text-muted-foreground xl:text-lg">
                    {stat.label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Projects */}
        <section id="projects" data-scroll-section>
          {/* Gradient */}
          <div className="relative isolate -z-10">
            <div
              className="absolute inset-x-0 -top-40 transform-gpu overflow-hidden blur-[100px] sm:-top-80 lg:-top-60"
              aria-hidden="true"
            >
              <div
                className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-primary via-primary to-secondary opacity-10 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]"
                style={{
                  clipPath:
                    "polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)",
                }}
              />
            </div>
          </div>
          <div data-scroll data-scroll-speed=".4" className="my-64">
            <span className="text-gradient clash-grotesk text-sm font-semibold tracking-tighter">
              ✨ Projects
            </span>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight tracking-tighter xl:text-6xl">
              Streamlined digital experiences.
            </h2>
            <p className="mt-1.5 text-base tracking-tight text-muted-foreground xl:text-lg">
              I&apos;ve worked on a variety of projects, from small websites to
              large-scale web applications. Here are some of my favorites:
            </p>

            {/* Carousel */}
            <div className="mt-14">
              <Carousel setApi={setCarouselApi} className="w-full">
                <CarouselContent>
                  {projects.map((project) => (
                    <CarouselItem key={project.title} className="md:basis-1/2">
                      <Card id="tilt">
                        <CardHeader className="p-0">
                          <Link href={project.href} target="_blank" passHref>
                            {project.image.endsWith(".webm") ? (
                              <video
                                src={project.image}
                                autoPlay
                                loop
                                muted
                                className="aspect-video h-full w-full rounded-t-md bg-primary object-cover"
                              />
                            ) : (
                              <Image
                                src={project.image}
                                alt={project.title}
                                width={600}
                                height={300}
                                quality={100}
                                className="aspect-video h-full w-full rounded-t-md bg-primary object-cover"
                              />
                            )}
                          </Link>
                        </CardHeader>
                        <CardContent className="absolute bottom-0 w-full bg-background/50 backdrop-blur">
                          <CardTitle className="border-t border-white/5 p-4 text-base font-normal tracking-tighter">
                            {project.description}
                          </CardTitle>
                        </CardContent>
                      </Card>
                    </CarouselItem>
                  ))}
                </CarouselContent>
                <CarouselPrevious />
                <CarouselNext />
              </Carousel>
              <div className="py-2 text-center text-sm text-muted-foreground">
                <span className="font-semibold">
                  {current} / {count}
                </span>{" "}
                projects
              </div>
            </div>
          </div>
        </section>

        {/* Detailed Projects Section */}
        <section id="project-details" data-scroll-section className="my-64">
          <div
            data-scroll
            data-scroll-speed=".4"
            data-scroll-position="top"
          >
            <span className="text-gradient clash-grotesk text-sm font-semibold tracking-tighter">
              💼 Featured Work
            </span>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight xl:text-6xl">
              Deep Dive into My Projects
            </h2>
            <p className="mt-1.5 max-w-3xl text-base tracking-tight text-muted-foreground xl:text-lg">
              Explore the technical details, challenges, and outcomes of my most impactful projects in AI, machine learning, and data science.
            </p>

            <div className="mt-14 space-y-16">
              {projectDetails.map((project, index) => (
                <motion.div
                  key={project.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="group relative"
                >
                  <Card className="overflow-hidden border-white/10 bg-white/5 backdrop-blur transition-all duration-300 hover:border-white/20 hover:bg-white/10">
                    <CardContent className="p-8">
                      <div className="flex flex-col gap-6 lg:flex-row lg:gap-8">
                        {/* Left Column - Main Info */}
                        <div className="flex-1 space-y-4">
                          <div>
                            <div className="mb-2 flex items-center gap-2">
                              <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                                {project.category}
                              </span>
                            </div>
                            <h3 className="text-2xl font-bold tracking-tight xl:text-3xl">
                              {project.title}
                            </h3>
                            <p className="mt-1 text-sm text-muted-foreground">
                              {project.description}
                            </p>
                          </div>

                          <p className="text-base leading-relaxed text-foreground/90">
                            {project.fullDescription}
                          </p>

                          {/* Technologies */}
                          <div>
                            <h4 className="mb-2 text-sm font-semibold text-foreground">
                              Technologies Used
                            </h4>
                            <div className="flex flex-wrap gap-2">
                              {project.technologies.map((tech) => (
                                <span
                                  key={tech}
                                  className="rounded-md bg-white/5 px-3 py-1 text-xs font-medium text-foreground backdrop-blur"
                                >
                                  {tech}
                                </span>
                              ))}
                            </div>
                          </div>

                          {/* Links */}
                          <div className="flex gap-3">
                            {project.github && project.github !== "#" && (
                              <Link href={project.github} target="_blank">
                                <Button variant="outline" size="sm">
                                  View Code
                                </Button>
                              </Link>
                            )}
                            {project.live && project.live !== "#" && (
                              <Link href={project.live} target="_blank">
                                <Button size="sm">
                                  Live Demo <ChevronRight className="ml-1 h-4 w-4" />
                                </Button>
                              </Link>
                            )}
                          </div>
                        </div>

                        {/* Right Column - Features & Metrics */}
                        <div className="space-y-6 lg:w-80">
                          {/* Key Features */}
                          <div>
                            <h4 className="mb-3 text-sm font-semibold text-foreground">
                              Key Features
                            </h4>
                            <ul className="space-y-2">
                              {project.features.map((feature, i) => (
                                <li
                                  key={i}
                                  className="flex items-start gap-2 text-sm text-muted-foreground"
                                >
                                  <ChevronRight className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" />
                                  <span>{feature}</span>
                                </li>
                              ))}
                            </ul>
                          </div>

                          {/* Metrics */}
                          {project.metrics && (
                            <div>
                              <h4 className="mb-3 text-sm font-semibold text-foreground">
                                Impact Metrics
                              </h4>
                              <div className="space-y-3">
                                {project.metrics.map((metric) => (
                                  <div
                                    key={metric.label}
                                    className="flex items-center justify-between rounded-lg bg-white/5 p-3"
                                  >
                                    <span className="text-xs text-muted-foreground">
                                      {metric.label}
                                    </span>
                                    <span className="text-lg font-bold text-gradient">
                                      {metric.value}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Research Experience Section */}
        <section id="research" data-scroll-section className="my-64">
          <div
            data-scroll
            data-scroll-speed=".4"
            data-scroll-position="top"
          >
            <span className="text-gradient clash-grotesk text-sm font-semibold tracking-tighter">
              🔬 Research
            </span>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight xl:text-6xl">
              Research Experience
            </h2>
            <p className="mt-1.5 max-w-3xl text-base tracking-tight text-muted-foreground xl:text-lg">
              Contributing to cutting-edge research in machine learning, NLP, and computer vision.
            </p>

            <div className="mt-14 space-y-8">
              {researchExperience.map((research, index) => (
                <motion.div
                  key={research.position}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card className="border-white/10 bg-white/5 backdrop-blur transition-all duration-300 hover:border-white/20 hover:bg-white/10">
                    <CardContent className="p-8">
                      <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                        <div className="flex-1">
                          <h3 className="text-2xl font-bold tracking-tight">
                            {research.position}
                          </h3>
                          <div className="mt-2 flex flex-col gap-1 text-sm text-muted-foreground">
                            <p className="font-semibold text-foreground">{research.institution}</p>
                            <p>{research.duration} • {research.location}</p>
                          </div>
                        </div>
                        <div className="text-sm text-muted-foreground lg:text-right">
                          <p className="font-medium text-primary">{research.impact}</p>
                        </div>
                      </div>

                      <p className="mt-4 text-base leading-relaxed text-foreground/90">
                        {research.description}
                      </p>

                      <div className="mt-6">
                        <h4 className="mb-3 text-sm font-semibold text-foreground">
                          Key Responsibilities
                        </h4>
                        <ul className="space-y-2">
                          {research.responsibilities.map((resp, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                              <ChevronRight className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" />
                              <span>{resp}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div className="mt-6 flex flex-wrap gap-2">
                        {research.technologies.map((tech) => (
                          <span
                            key={tech}
                            className="rounded-md bg-white/5 px-3 py-1 text-xs font-medium text-foreground backdrop-blur"
                          >
                            {tech}
                          </span>
                        ))}
                      </div>

                      {research.publications.length > 0 && (
                        <div className="mt-6">
                          <h4 className="mb-2 text-sm font-semibold text-foreground">
                            Related Publications
                          </h4>
                          <ul className="space-y-1">
                            {research.publications.map((pub, i) => (
                              <li key={i} className="text-sm text-muted-foreground">
                                • {pub}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Internships Section */}
        <section id="internships" data-scroll-section className="my-64">
          <div
            data-scroll
            data-scroll-speed=".4"
            data-scroll-position="top"
          >
            <span className="text-gradient clash-grotesk text-sm font-semibold tracking-tighter">
              💼 Experience
            </span>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight xl:text-6xl">
              Professional Internships
            </h2>
            <p className="mt-1.5 max-w-3xl text-base tracking-tight text-muted-foreground xl:text-lg">
              Hands-on industry experience building production-ready ML systems and data-driven solutions.
            </p>

            <div className="mt-14 space-y-8">
              {internships.map((internship, index) => (
                <motion.div
                  key={internship.role}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card className="border-white/10 bg-white/5 backdrop-blur transition-all duration-300 hover:border-white/20 hover:bg-white/10">
                    <CardContent className="p-8">
                      <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                        <div className="flex-1">
                          <div className="mb-2 flex items-center gap-2">
                            <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                              {internship.type}
                            </span>
                          </div>
                          <h3 className="text-2xl font-bold tracking-tight">
                            {internship.role}
                          </h3>
                          <div className="mt-2 flex flex-col gap-1 text-sm text-muted-foreground">
                            <p className="font-semibold text-foreground">{internship.company}</p>
                            <p>{internship.duration} • {internship.location}</p>
                          </div>
                        </div>
                        <div className="text-sm text-muted-foreground lg:text-right">
                          <p className="font-medium text-primary">{internship.impact}</p>
                        </div>
                      </div>

                      <p className="mt-4 text-base leading-relaxed text-foreground/90">
                        {internship.description}
                      </p>

                      <div className="mt-6">
                        <h4 className="mb-3 text-sm font-semibold text-foreground">
                          Key Achievements
                        </h4>
                        <ul className="space-y-2">
                          {internship.achievements.map((achievement, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                              <ChevronRight className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" />
                              <span>{achievement}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div className="mt-6 grid gap-6 md:grid-cols-2">
                        <div>
                          <h4 className="mb-2 text-sm font-semibold text-foreground">
                            Technologies
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {internship.technologies.map((tech) => (
                              <span
                                key={tech}
                                className="rounded-md bg-white/5 px-3 py-1 text-xs font-medium text-foreground backdrop-blur"
                              >
                                {tech}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div>
                          <h4 className="mb-2 text-sm font-semibold text-foreground">
                            Skills Developed
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {internship.skills.map((skill) => (
                              <span
                                key={skill}
                                className="rounded-md bg-primary/10 px-3 py-1 text-xs font-medium text-primary backdrop-blur"
                              >
                                {skill}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Teaching Experience Section */}
        <section id="teaching" data-scroll-section className="my-64">
          <div
            data-scroll
            data-scroll-speed=".4"
            data-scroll-position="top"
          >
            <span className="text-gradient clash-grotesk text-sm font-semibold tracking-tighter">
              👨‍🏫 Teaching
            </span>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight xl:text-6xl">
              Teaching Experience
            </h2>
            <p className="mt-1.5 max-w-3xl text-base tracking-tight text-muted-foreground xl:text-lg">
              Supporting graduate education in data science, big data engineering, and prescriptive analytics.
            </p>

            <div className="mt-14 space-y-8">
              {teachingExperience.map((teaching, index) => (
                <motion.div
                  key={teaching.position}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card className="border-white/10 bg-white/5 backdrop-blur transition-all duration-300 hover:border-white/20 hover:bg-white/10">
                    <CardContent className="p-8">
                      <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                        <div className="flex-1">
                          <div className="mb-2 flex items-center gap-2">
                            <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                              {teaching.type}
                            </span>
                          </div>
                          <h3 className="text-2xl font-bold tracking-tight">
                            {teaching.position}
                          </h3>
                          <div className="mt-2 flex flex-col gap-1 text-sm text-muted-foreground">
                            <p className="font-semibold text-foreground">{teaching.institution}</p>
                            <p>{teaching.duration} • {teaching.location}</p>
                          </div>
                        </div>
                        <div className="text-sm text-muted-foreground lg:text-right">
                          <p className="font-medium text-primary">{teaching.impact}</p>
                        </div>
                      </div>

                      {/* Courses Taught */}
                      <div className="mt-6 space-y-6">
                        {teaching.courses.map((course, idx) => (
                          <div key={course.code} className="rounded-lg bg-white/5 p-6">
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <h4 className="text-lg font-bold text-foreground">
                                  {course.code} - {course.name}
                                </h4>
                                <p className="mt-1 text-sm text-muted-foreground">{course.duration}</p>
                              </div>
                            </div>
                            
                            <div className="mt-4">
                              <h5 className="mb-2 text-sm font-semibold text-foreground">
                                Course Topics Covered:
                              </h5>
                              <ul className="space-y-2">
                                {course.topics.map((topic, i) => (
                                  <li
                                    key={i}
                                    className="flex items-start gap-2 text-sm text-muted-foreground"
                                  >
                                    <ChevronRight className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" />
                                    <span>{topic}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Responsibilities */}
                      <div className="mt-6">
                        <h4 className="mb-3 text-sm font-semibold text-foreground">
                          Key Responsibilities
                        </h4>
                        <ul className="space-y-2">
                          {teaching.responsibilities.map((resp, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                              <ChevronRight className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" />
                              <span>{resp}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Technologies */}
                      <div className="mt-6">
                        <h4 className="mb-2 text-sm font-semibold text-foreground">
                          Technologies & Tools
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {teaching.technologies.map((tech) => (
                            <span
                              key={tech}
                              className="rounded-md bg-white/5 px-3 py-1 text-xs font-medium text-foreground backdrop-blur"
                            >
                              {tech}
                            </span>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Education Section */}
        <section id="education" data-scroll-section className="my-64">
          <div
            data-scroll
            data-scroll-speed=".4"
            data-scroll-position="top"
          >
            <span className="text-gradient clash-grotesk text-sm font-semibold tracking-tighter">
              🎓 Education
            </span>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight xl:text-6xl">
              Academic Journey
            </h2>
            <p className="mt-1.5 max-w-3xl text-base tracking-tight text-muted-foreground xl:text-lg">
              Building a strong foundation in computer science and specializing in AI/ML through rigorous coursework and research.
            </p>

            <div className="mt-14 space-y-8">
              {education.map((edu, index) => (
                <motion.div
                  key={edu.degree}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card className="border-white/10 bg-white/5 backdrop-blur transition-all duration-300 hover:border-white/20 hover:bg-white/10">
                    <CardContent className="p-8">
                      <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                        <div className="flex-1">
                          <h3 className="text-2xl font-bold tracking-tight">
                            {edu.degree}
                          </h3>
                          <div className="mt-2 flex flex-col gap-1 text-sm text-muted-foreground">
                            <p className="font-semibold text-foreground">{edu.institution}</p>
                            <p>{edu.duration} • {edu.location}</p>
                            <p className="mt-1">
                              <span className="font-semibold text-primary">Focus:</span> {edu.focus}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-3xl font-bold text-gradient">
                            {edu.gpa}
                          </div>
                          <p className="text-sm text-muted-foreground">GPA</p>
                        </div>
                      </div>

                      {edu.thesis && (
                        <div className="mt-4">
                          <p className="text-sm text-muted-foreground">
                            <span className="font-semibold text-foreground">Thesis:</span> {edu.thesis}
                          </p>
                        </div>
                      )}

                      <div className="mt-6 grid gap-6 md:grid-cols-2">
                        <div>
                          <h4 className="mb-3 text-sm font-semibold text-foreground">
                            Relevant Coursework
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {edu.coursework.map((course) => (
                              <span
                                key={course}
                                className="rounded-md bg-white/5 px-3 py-1 text-xs font-medium text-foreground backdrop-blur"
                              >
                                {course}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div>
                          <h4 className="mb-3 text-sm font-semibold text-foreground">
                            Achievements & Honors
                          </h4>
                          <ul className="space-y-1">
                            {edu.achievements.map((achievement, i) => (
                              <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                                <ChevronRight className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" />
                                <span>{achievement}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Achievements & Publications Section */}
        <section id="achievements" data-scroll-section className="my-64">
          <div
            data-scroll
            data-scroll-speed=".4"
            data-scroll-position="top"
          >
            <span className="text-gradient clash-grotesk text-sm font-semibold tracking-tighter">
              🏆 Recognition
            </span>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight xl:text-6xl">
              Achievements & Publications
            </h2>
            <p className="mt-1.5 max-w-3xl text-base tracking-tight text-muted-foreground xl:text-lg">
              Recognition for research contributions, competitive achievements, and academic excellence.
            </p>

            <div className="mt-14 space-y-12">
              {achievements.map((section, sectionIndex) => (
                <motion.div
                  key={section.category}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: sectionIndex * 0.1 }}
                  viewport={{ once: true }}
                >
                  <h3 className="mb-6 text-2xl font-bold tracking-tight">
                    {section.category}
                  </h3>

                  <div className="space-y-6">
                    {section.items.map((item, itemIndex) => (
                      <Card
                        key={itemIndex}
                        className="border-white/10 bg-white/5 backdrop-blur transition-all duration-300 hover:border-white/20 hover:bg-white/10"
                      >
                        <CardContent className="p-6">
                          {section.category === "Publications" ? (
                            <>
                              <div className="flex items-start justify-between gap-4">
                                <div className="flex-1">
                                  <div className="mb-2 flex flex-wrap items-center gap-2">
                                    <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                                      {item.type}
                                    </span>
                                    <span className={`rounded-full px-3 py-1 text-xs font-medium ${
                                      item.status === "Published" 
                                        ? "bg-green-500/10 text-green-500" 
                                        : "bg-yellow-500/10 text-yellow-500"
                                    }`}>
                                      {item.status}
                                    </span>
                                  </div>
                                  <h4 className="text-lg font-bold tracking-tight">
                                    {item.title}
                                  </h4>
                                  <p className="mt-1 text-sm text-muted-foreground">
                                    {item.authors}
                                  </p>
                                  <p className="mt-1 text-sm font-medium text-foreground">
                                    {item.venue} • {item.year}
                                  </p>
                                </div>
                                {item.citation && (
                                  <div className="text-right">
                                    <p className="text-sm font-semibold text-primary">
                                      {item.citation}
                                    </p>
                                  </div>
                                )}
                              </div>
                              <p className="mt-3 text-sm leading-relaxed text-foreground/90">
                                {item.description}
                              </p>
                              {item.link && item.link !== "#" && (
                                <div className="mt-4">
                                  <Link href={item.link} target="_blank">
                                    <Button variant="outline" size="sm">
                                      View Paper <ChevronRight className="ml-1 h-4 w-4" />
                                    </Button>
                                  </Link>
                                </div>
                              )}
                            </>
                          ) : section.category === "Patents & IP" ? (
                            <>
                              <div className="flex items-start justify-between gap-4">
                                <div className="flex-1">
                                  <div className="mb-2">
                                    <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                                      {item.status}
                                    </span>
                                  </div>
                                  <h4 className="text-lg font-bold tracking-tight">
                                    {item.title}
                                  </h4>
                                  <p className="mt-1 text-sm text-muted-foreground">
                                    {item.number} • {item.year}
                                  </p>
                                </div>
                              </div>
                              <p className="mt-3 text-sm leading-relaxed text-foreground/90">
                                {item.description}
                              </p>
                            </>
                          ) : section.category === "Certifications" ? (
                            <>
                              <div className="flex items-start justify-between gap-4">
                                <div className="flex-1">
                                  <h4 className="text-lg font-bold tracking-tight">
                                    {item.title}
                                  </h4>
                                  <p className="mt-1 text-sm text-muted-foreground">
                                    {item.organization} • {item.year}
                                  </p>
                                </div>
                              </div>
                              {item.link && (
                                <div className="mt-3">
                                  <Link href={item.link} target="_blank">
                                    <Button variant="outline" size="sm">
                                      View Credential <ChevronRight className="ml-1 h-4 w-4" />
                                    </Button>
                                  </Link>
                                </div>
                              )}
                            </>
                          ) : (
                            <>
                              <div className="flex items-start justify-between gap-4">
                                <div className="flex-1">
                                  <h4 className="text-lg font-bold tracking-tight">
                                    {item.title}
                                  </h4>
                                  <p className="mt-1 text-sm font-medium text-foreground">
                                    {item.organization} • {item.year}
                                  </p>
                                </div>
                              </div>
                              <p className="mt-3 text-sm leading-relaxed text-foreground/90">
                                {item.description}
                              </p>
                            </>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Services */}
        <section id="services" data-scroll-section>
          <div
            data-scroll
            data-scroll-speed=".4"
            data-scroll-position="top"
            className="my-24 flex flex-col justify-start space-y-10"
          >
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{
                duration: 1,
                staggerChildren: 0.5,
              }}
              viewport={{ once: true }}
              className="grid items-center gap-1.5 md:grid-cols-2 xl:grid-cols-3"
            >
              <div className="flex flex-col py-6 xl:p-6">
                <h2 className="text-4xl font-medium tracking-tight">
                  Need more info?
                  <br />
                  <span className="text-gradient clash-grotesk tracking-normal">
                    I got you.
                  </span>
                </h2>
                <p className="mt-2 tracking-tighter text-secondary-foreground">
                  Here are some of the services I offer. If you have any
                  questions, feel free to reach out.
                </p>
              </div>
              {services.map((service) => (
                <div
                  key={service.service}
                  className="flex h-[320px] flex-col items-start rounded-md bg-white/5 p-14 shadow-md backdrop-blur transition duration-300 hover:-translate-y-0.5 hover:bg-white/10 hover:shadow-md"
                >
                  <service.icon className="my-6 text-primary" size={20} />
                  <span className="text-lg tracking-tight text-foreground">
                    {service.service}
                  </span>
                  <span className="mt-2 tracking-tighter text-muted-foreground">
                    {service.description}
                  </span>
                </div>
              ))}
            </motion.div>
          </div>
        </section>

        {/* Contact */}
        <section id="contact" data-scroll-section className="my-64">
          <div
            data-scroll
            data-scroll-speed=".4"
            data-scroll-position="top"
            className="flex flex-col items-center justify-center rounded-lg bg-gradient-to-br from-primary/[6.5%] to-white/5 px-8 py-16 text-center xl:py-24"
          >
            <h2 className="text-4xl font-medium tracking-tighter xl:text-6xl">
              Let&apos;s work{" "}
              <span className="text-gradient clash-grotesk">together.</span>
            </h2>
            <p className="mt-1.5 text-base tracking-tight text-muted-foreground xl:text-lg">
              I&apos;m currently available for freelance work and open to
              discussing new projects.
            </p>
            <Link href="mailto:ftkiranraj@proton.me" passHref>
              <Button className="mt-6">Get in touch</Button>
            </Link>
          </div>
        </section>
      </div>
    </Container>
  );
}

function Gradient() {
  return (
    <>
      {/* Upper gradient */}
      <div className="absolute -top-40 right-0 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
        <svg
          className="relative left-[calc(50%-11rem)] -z-10 h-[21.1875rem] max-w-none -translate-x-1/2 rotate-[30deg] sm:left-[calc(50%-30rem)] sm:h-[42.375rem]"
          viewBox="0 0 1155 678"
        >
          <path
            fill="url(#45de2b6b-92d5-4d68-a6a0-9b9b2abad533)"
            fillOpacity=".1"
            d="M317.219 518.975L203.852 678 0 438.341l317.219 80.634 204.172-286.402c1.307 132.337 45.083 346.658 209.733 145.248C936.936 126.058 882.053-94.234 1031.02 41.331c119.18 108.451 130.68 295.337 121.53 375.223L855 299l21.173 362.054-558.954-142.079z"
          />
          <defs>
            <linearGradient
              id="45de2b6b-92d5-4d68-a6a0-9b9b2abad533"
              x1="1155.49"
              x2="-78.208"
              y1=".177"
              y2="474.645"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#7980fe" />
              <stop offset={1} stopColor="#f0fff7" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      {/* Lower gradient */}
      <div className="absolute inset-x-0 top-[calc(100%-13rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-30rem)]">
        <svg
          className="relative left-[calc(50%+3rem)] h-[21.1875rem] max-w-none -translate-x-1/2 sm:left-[calc(50%+36rem)] sm:h-[42.375rem]"
          viewBox="0 0 1155 678"
        >
          <path
            fill="url(#ecb5b0c9-546c-4772-8c71-4d3f06d544bc)"
            fillOpacity=".1"
            d="M317.219 518.975L203.852 678 0 438.341l317.219 80.634 204.172-286.402c1.307 132.337 45.083 346.658 209.733 145.248C936.936 126.058 882.053-94.234 1031.02 41.331c119.18 108.451 130.68 295.337 121.53 375.223L855 299l21.173 362.054-558.954-142.079z"
          />
          <defs>
            <linearGradient
              id="ecb5b0c9-546c-4772-8c71-4d3f06d544bc"
              x1="1155.49"
              x2="-78.208"
              y1=".177"
              y2="474.645"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#9A70FF" />
              <stop offset={1} stopColor="#838aff" />
            </linearGradient>
          </defs>
        </svg>
      </div>
    </>
  );
}
