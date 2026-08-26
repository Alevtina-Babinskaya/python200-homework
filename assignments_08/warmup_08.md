Cloud Concepts Question 1 
What is the core economic model of cloud computing, and how does it differ from owning your own servers?
Complex computing processes require extensive resources that can be expensive and time-consuming to build. Cloud services allow users to rent and use resources for computing and storing data, which can be cheaper, easier to scale, and more reliable.

Cloud Concepts Question 2
What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.
Vertical scaling is increasing the power of an existing resource, for example, adding storage or RAM to a PC. Horizontal scaling is adding another resource to split the work. Horizontal scaling is possible when the process can be split into parallel branches. Vertical scaling might be cheaper in some cases.
Examples: A small business has a database server that is running out of memory, so they upgrade the server by adding more RAM and a faster CPU - vertical scaling. 
A company needs to process thousands of images. Since each image can be processed independently, they can add more machines to split the work - horizontal scaling.  
Horizontal: A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
Vertical: A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
Horizontal: A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

Cloud Concepts Question 3
Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.
Gmail - SaaS
Azure Virtual Machines - IaaS
AWS S3 (Simple Storage Service) - IaaS
GitHub Codespaces - PaaS
Snowflake - SaaS
Supabase - PaaS
Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.
IaaS — Infrastructure as a Service — the cloud provides raw resources: machines, storage, and networking. I am responsible for installing my software, setting up my environment, and managing security and other configurations. Example: Google Compute Engine.
PaaS — Platform as a Service — the cloud provides a platform for coding, running, and deploying apps. I bring my code and deploy my application without managing the underlying infrastructure. Example: Google App Engine.
SaaS — Software as a Service — the cloud provides ready-to-use software. I don't manage the underlying infrastructure or software; I just use the application. Example: Google Docs.

Cloud Concepts Question 4
What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?
Managed data platforms are a layer on top of cloud providers. They are optimized specifically for data analysis and are easier and faster to get started with, but they offer less flexibility and can cost more than using raw cloud infrastructure.

Cloud Concepts Question 5
The lesson names two situations where the cloud is probably not the right choice. What are they?
If dataset can be set on a single machine or computation don't demand much power, local processing is faster, cheaper, and easier to fullfill and support.

Cloud Landscape Question 1
Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.
- Amazon Web Services (AWS) is the oldest major cloud provider and holds over a third of the cloud market. Large enterprises are most likely to work with AWS.
- Google Cloud Platform (GCP) is strongest in data analytics and machine learning. It is a good choice for companies working with big data and ML infrastructure.
- Microsoft Azure is popular because of its integration with Windows. Non-profit and government organizations often use it through existing agreements with Microsoft.

Cloud Landscape Question 2
The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.
Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?
- Access. Azure requires a tenant invitation to join, while Supabase access can be set up in minutes.
- Pedagogical fit. Azure stores data as files organized by paths, while Supabase stores data in rows and columns, which is better for educational goals because it makes filtering, searching, and querying easier and is similar to databases students may already know.
- Pipeline coherence. In Supabase, raw and enriched data are stored in two separate tables, which follows good data analysis practices.

I think that when starting a new project, we should decide what features are important for the project and whether the cloud tool matches our requirements. Then, we should consider how much time we have to spend setting up the environment and how much we might pay for using the cloud service. The tool should be convenient and consistent with the other tools we need to reach our goal.

Cloud Landscape Question 3
For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

Object storage (S3, Supabase Storage): You need to store 10 TB of image files and retrieve them by filename from any machine.
ML platform (Azure ML): You need to run an ML training job on a GPU for four hours, then shut it down.
Serverless compute (Lambda): You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
LLM API (Bedrock): You need to send structured data to a large language model and get a text response back.

Cloud Landscape Question 4
The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?
I am creating an online store and use Amazon S3 for storing data and Google BigQuery to analyze sales statistics. Using different providers allows me to choose the best service for each task. Consolidating everything to one provider could make the project easier to manage and integrate.