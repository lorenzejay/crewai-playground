from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

transcript_analyzer = Agent(
    role="Analyze the overall structure and flow of the conversation: {transcript}",
    goal="Extract key points and summarize the interaction",
    backstory="Expert in conversation analysis and information extraction",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

sentiment_analyzer = Agent(
    role="Analyze the emotional tone of both the customer and the support rep: {transcript}",
    goal="Identify changes in sentiment throughout the call and overall satisfaction",
    backstory="Specialized in natural language processing and emotion detection",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

issue_categorizer = Agent(
    role="Categorize the main issue(s) discussed in the call: {transcript}",
    goal="Identify and classify the primary and any secondary issues raised",
    backstory="Experienced in taxonomy creation and issue classification systems",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

resolution_efficiency = Agent(
    role="Evaluate the efficiency of issue resolution: {transcript}",
    goal="Assess how quickly and effectively the issue was resolved",
    backstory="Expert in customer service metrics and efficiency optimization",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

quality_assurance = Agent(
    role="Evaluate the overall quality of the customer service interaction: {transcript}",
    goal="Identify areas of excellence and opportunities for improvement",
    backstory="Seasoned in customer service best practices and training",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

report_generator = Agent(
    role="Aggregate insights and generate comprehensive report: {transcript}",
    goal="Create a clear, actionable report for managers based on call analysis",
    backstory="Experienced data analyst skilled in synthesizing information and creating executive summaries",
    llm=llm,
    verbose=True,
)

# Define the tasks with expected outputs

analyze_transcript_task = Task(
    description="Analyze the transcript and extract key points: {transcript}",
    agent=transcript_analyzer,
    expected_output="A summary of the conversation including main topics discussed, any problems identified, and solutions provided.",
    async_execution=True,
)

analyze_sentiment_task = Task(
    description="Perform sentiment analysis on the conversation: {transcript}",
    agent=sentiment_analyzer,
    expected_output="A breakdown of sentiment for both customer and representative, including any shifts in emotion throughout the call and an overall sentiment score.",
    async_execution=True,
)

analyze_issue_task = Task(
    description="Categorize the main issue(s) discussed in the call: {transcript}",
    agent=issue_categorizer,
    expected_output="A list of identified issues categorized by type (e.g., technical, billing, product information) and their priority level.",
    async_execution=True,
)

analyze_resolution_efficiency_task = Task(
    description="Evaluate the efficiency of issue resolution: {transcript}",
    agent=resolution_efficiency,
    expected_output="Metrics on resolution time, number of steps taken, and any escalations required. Include a rating of resolution efficiency (e.g., 1-10 scale).",
    async_execution=True,
)

analyze_quality_task = Task(
    description="Assess overall quality and provide improvement suggestions: {transcript}",
    agent=quality_assurance,
    expected_output="A quality score (e.g., 1-10) with detailed feedback on representative performance, adherence to protocols, and specific areas for improvement.",
    async_execution=True,
)

generate_report_task = Task(
    description="Aggregate all insights and generate a comprehensive report",
    agent=report_generator,
    expected_output="A structured report including an executive summary, key findings, detailed analysis, recommendations, and next steps, suitable for management review.",
    context=[
        analyze_transcript_task,
        analyze_sentiment_task,
        analyze_issue_task,
        analyze_resolution_efficiency_task,
        analyze_quality_task,
    ],
    output_file="report.md",
)

# Create the crew
call_analytics_crew = Crew(
    agents=[
        transcript_analyzer,
        sentiment_analyzer,
        issue_categorizer,
        resolution_efficiency,
        quality_assurance,
        report_generator,
    ],
    tasks=[
        analyze_transcript_task,
        analyze_sentiment_task,
        analyze_issue_task,
        analyze_resolution_efficiency_task,
        analyze_quality_task,
        generate_report_task,
    ],
    verbose=2,
)


transcript = """
Rep: Thank you for calling ACME Support. My name is Sarah. How may I assist you today?

Customer: Hi Sarah, I'm having some issues with my iPhone. It's been acting really strange lately.

Rep: I'm sorry to hear that you're experiencing problems with your iPhone. I'd be happy to help you troubleshoot. Could you please provide me with your name and the model of your iPhone?

Customer: Sure, my name is Alex, and I have an iPhone 12.

Rep: Thank you, Alex. To better assist you, could you describe the strange behavior you're noticing with your iPhone 12?

Customer: Well, it's a few things. The battery seems to drain really fast, some of my apps keep crashing, and sometimes the phone feels hot to the touch.

Rep: I understand, Alex. Those issues can certainly be frustrating. Let's go through them one by one. First, can you tell me which version of iOS your iPhone is currently running?

Customer: Um, I'm not sure. How do I check that?

Rep: No problem, I can guide you. Please go to your Settings app, then tap on "General," and then "About." The software version should be listed there.

Customer: Okay, I see it. It says iOS 15.4.1.

Rep: Thank you for checking. It looks like your iPhone isn't on the latest version of iOS. Updating your software might resolve some of these issues. Are you comfortable with updating your iPhone now?

Customer: Sure, that sounds good. How do I do that?

Rep: Great! Before we start the update, let's make sure your iPhone is sufficiently charged and connected to Wi-Fi. Can you confirm that for me?

Customer: Yes, I'm at 68% battery and connected to my home Wi-Fi.

Rep: Perfect. Now, go back to the main Settings page, tap on "General," and then "Software Update." Let me know what you see there.

Customer: It says "iOS 16.5 is available for download and install."

Rep: Excellent. Go ahead and tap "Download and Install." You'll need to enter your passcode to proceed. The download might take a few minutes, depending on your internet speed.

Customer: Okay, it's downloading now.

Rep: While that's downloading, let's discuss the app crashing issue. Can you tell me which apps are having problems?

Customer: It's mainly my social media apps like Instagram and Twitter, and sometimes my banking app.

Rep: I see. Once the update is complete, that might resolve the app crashing issues. However, if it persists, we may need to look at updating or reinstalling those specific apps. 

Customer: Alright, that makes sense.

Rep: Now, regarding the battery drain and the phone feeling hot, these can often be related. Are there any specific times when you notice the phone getting particularly warm?

Customer: Yeah, it seems to happen when I'm using GPS for navigation or playing graphics-intensive games.

Rep: That's actually quite normal. GPS and gaming can be demanding on the processor, which can cause the phone to warm up and use more battery. However, if it's uncomfortably hot or you notice it in other situations, that could indicate a problem.

Customer: I see. Is there anything I can do to help with the battery life?

Rep: Absolutely. Once the update is complete, go to Settings, then Battery, and check the Battery Health. Also, look at your Battery Usage by App to see if any apps are using an unusual amount of power in the background.

Customer: Okay, the update has finished downloading and it's asking me to install now.

Rep: Great! Go ahead and start the installation. Your phone will restart during this process, which is normal. It should take about 10-15 minutes.

Customer: Alright, it's installing now.

Rep: Perfect. While we wait, let me give you a few more tips for optimizing your battery life. Lowering your screen brightness, turning off background app refresh for apps you don't need updated constantly, and using Wi-Fi instead of cellular data when possible can all help extend battery life.

Customer: Those are helpful tips, thank you.

Rep: You're welcome. Also, regarding the app crashes, if they continue after the update, try deleting and reinstalling the problematic apps. If that doesn't work, we can look into resetting your app data or even restoring your iPhone as a last resort.

Customer: Got it. The phone has restarted and... it looks like it's completed the update!

Rep: Excellent! Can you please check the software version again to confirm the update was successful?

Customer: Yes, it now says iOS 16.5.

Rep: Perfect. Now, let's check that Battery Health we talked about earlier.

Customer: Okay, I'm in Battery Health and it says "Maximum Capacity: 92%". Is that good?

Rep: Yes, that's quite good. It means your battery is still able to hold 92% of its original capacity. Now, let's give your phone a day or two to adjust to the new software. Often, battery life can seem worse right after an update as the phone reindexes, but it should improve.

Customer: That's good to know. What should I do if I'm still having issues after a couple of days?

Rep: If you're still experiencing problems, please don't hesitate to call us back. We can then look into more advanced troubleshooting steps or possibly schedule a hardware diagnostic if necessary.

Customer: Okay, thank you so much for your help, Sarah. I really appreciate it.

Rep: You're very welcome, Alex. Is there anything else I can assist you with today?

Customer: No, I think that covers everything. Thanks again!

Rep: It was my pleasure to help. Thank you for choosing ACME Support. Have a great day!

Customer: You too, goodbye!

Rep: Goodbye!
"""


result = call_analytics_crew.kickoff(inputs={"transcript": transcript})
print("result: ", result)
