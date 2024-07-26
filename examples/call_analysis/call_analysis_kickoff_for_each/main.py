import asyncio
from crewai import Agent, Task, Crew
from crewai_tools import TXTSearchTool

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

transcript_analyzer = Agent(
    role="Transcript Analyzer",
    goal="""
        Analyze the overall structure and flow of the conversation on topic: '{topic}'
        Extract key points and summarize the interaction in the transcript file: '{transcript}'
    """,
    backstory="Expert in conversation analysis and information extraction",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    tools=[TXTSearchTool()],
    max_iter=10,
)

sentiment_analyzer = Agent(
    role="Sentiment Analyzer",
    goal="""
        Analyze the emotional tone of both the customer and the support rep on topic: '{topic}'
        Identify changes in sentiment throughout the call and overall satisfaction from the transcript file: '{transcript}'
    """,
    backstory="Specialized in natural language processing and emotion detection",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    tools=[TXTSearchTool()],
    max_iter=10,
)

issue_categorizer = Agent(
    role="Issue Categorizer",
    goal="""
        Categorize the main issue(s) discussed on topic: '{topic}'
        Identify and classify the primary and any secondary issues raised from the transcript file: '{transcript}'
    """,
    backstory="Experienced in taxonomy creation and issue classification systems",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    tools=[TXTSearchTool()],
    max_iter=10,
)


quality_assurance = Agent(
    role="Quality Assurance",
    goal="""
        Evaluate the overall quality of the customer service interaction on topic: '{topic}'
        Identify areas of excellence and opportunities for improvement from the transcript file: '{transcript}'
    """,
    backstory="Seasoned in customer service best practices and training",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    tools=[TXTSearchTool()],
    max_iter=10,
)

report_generator = Agent(
    role="Report Generator",
    goal="""
        Create a clear, actionable report for managers based on call analysis from the transcript files.
        The report should include the names of the customer and customer service rep, their quality score, executive summary, key findings, detailed analysis, recommendations, and next steps, suitable for management review.
    """,
    backstory="Experienced data analyst skilled in synthesizing information and creating executive summaries",
    llm=llm,
    verbose=True,
)


analyze_transcript_task = Task(
    description="Analyze the transcript on topic '{topic}' and extract key points: '{transcript}'",
    agent=transcript_analyzer,
    expected_output="A summary of the conversation including main topics discussed, any problems identified, and solutions provided.",
    async_execution=True,
)

analyze_sentiment_task = Task(
    description="Perform sentiment analysis on the conversation on topic '{topic}' in the transcript file: '{transcript}'",
    agent=sentiment_analyzer,
    expected_output="A breakdown of sentiment for both customer and representative, including any shifts in emotion throughout the call and an overall sentiment score.",
    async_execution=True,
)

analyze_issue_task = Task(
    description="Categorize the main issue(s) discussed on topic '{topic}' in the transcript file: '{transcript}'",
    agent=issue_categorizer,
    expected_output="A list of identified issues categorized by type (e.g., technical, billing, product information) and their priority level.",
    async_execution=True,
)


analyze_quality_task = Task(
    description="Assess overall quality and provide improvement suggestions on topic '{topic}' in the transcript file: '{transcript}'",
    agent=quality_assurance,
    expected_output="A quality score (e.g., 1-10) with detailed feedback on representative performance, adherence to protocols, and specific areas for improvement.",
    async_execution=True,
)

generate_report_task = Task(
    description="Aggregate all insights and generate a comprehensive report",
    agent=report_generator,
    expected_output="A structured report including an the names of the customer and customer serivce rep, their quality score, executive summary, key findings, detailed analysis, recommendations, and next steps, suitable for management review.",
    context=[
        analyze_transcript_task,
        analyze_sentiment_task,
        analyze_issue_task,
        analyze_quality_task,
    ],
    output_file="{report_output_file}",
)

# Create the crew
call_analytics_crew = Crew(
    agents=[
        transcript_analyzer,
        sentiment_analyzer,
        issue_categorizer,
        quality_assurance,
        report_generator,
    ],
    tasks=[
        analyze_transcript_task,
        analyze_sentiment_task,
        analyze_issue_task,
        analyze_quality_task,
        generate_report_task,
    ],
    verbose=2,
)

transcripts = [
    {
        "transcript": "transcripts/cs_call_1.txt",
        "report_output_file": "reports_generated/report_1.txt",
        "topic": "iPhone issues",
    },
    {
        "transcript": "transcripts/cs_call_2.txt",
        "report_output_file": "reports_generated/report_2.txt",
        "topic": "Dell Laptop issues",
    },
]


async def main():
    async_results = await call_analytics_crew.kickoff_for_each_async(inputs=transcripts)
    for async_result in async_results:
        print("async_result", async_result)


if __name__ == "__main__":
    asyncio.run(main())
