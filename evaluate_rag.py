import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
import traceback

# Import your RAG query function
from query_data import enhanced_process_query

def evaluate_rag_system():
    """
    Evaluate RAG system by comparing its answers to ground truth answers.
    Calculates cosine similarity between generated and ground truth answers.
    Saves results to an Excel file with detailed statistics.
    """
    # Path for results
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f"rag_evaluation_{timestamp}.xlsx")
    
    # Initialize the embedding model for similarity calculation
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Define all questions and ground truth answers
    print("Initializing questions and ground truth answers...")
    questions_answers = [
    # ASK ME ANYTHING PDF
    {
        "pdf": "ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf",
        "question_type": "Summarization of PDF",
        "question": "What is the core strategy proposed in the paper for improving the reliability of language model outputs without relying on a single perfect prompt?",
        "ground_truth": "The paper proposes Ask Me Anything Prompting (AMA), a strategy that uses multiple imperfect prompts in a question-answering format and aggregates their outputs using weak supervision to improve prediction reliability and performance across tasks and models (p. 1–2)."
    },
    {
        "pdf": "ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf",
        "question_type": "Question Related to Any Image in PDF",
        "question": "How does the paper illustrate the transformation of task inputs into question-answer formats and the subsequent aggregation of responses?",
        "ground_truth": "The paper shows that task inputs are reformatted into effective QA prompts using prompt-chains, where the model first generates a question from a statement and then answers it. These outputs are aggregated using weak supervision to produce a final prediction (p. 2, described in the caption of Figure 1)."
    },
    {
        "pdf": "ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf",
        "question_type": "Question Related to Any Table in PDF",
        "question": "What evidence is presented to show that a smaller open-source language model can outperform a much larger model on several benchmarks?",
        "ground_truth": "The GPT-J-6B model using AMA outperforms the few-shot GPT-3 175B model on 15 of 20 benchmarks, with an average improvement of 41% over the 6B model's few-shot baseline (p. 8, Table 1)."
    },
    {
        "pdf": "ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf",
        "question_type": "Question Related to Any Text in PDF",
        "question": "Why do the authors argue that open-ended question formats are generally more effective than restrictive prompts in language model performance?",
        "ground_truth": "Open-ended prompts align better with the language modeling objective and appear more frequently in pretraining data, leading to better performance. For example, converting restrictive prompts to QA formats improved performance by 72% on average across three SuperGLUE tasks (p. 4–5)."
    },
    {
        "pdf": "ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf",
        "question_type": "Questions from Multiple Pages in PDF (2–3 Adjacent Pages)",
        "question": "How does the paper develop the AMA strategy across the sections discussing prompt effectiveness, scalable prompt generation, and aggregation?",
        "ground_truth": "The paper first identifies that QA prompts are more effective (Section 3.2), then introduces a scalable method using prompt-chains to generate these prompts (Section 3.3), and finally proposes weak supervision to aggregate the outputs, accounting for prompt dependencies and varying accuracies (Section 3.4, p. 4–6)."
    },
    {
        "pdf": "ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf",
        "question_type": "Questions from Start and End of PDF",
        "question": "How does the introduction frame the problem of prompt brittleness, and what conclusions are drawn about the broader implications of the AMA strategy?",
        "ground_truth": "The introduction highlights that prompt design is brittle and labor-intensive. The conclusion emphasizes that AMA enables smaller, open-source models to match or exceed large proprietary models, making LLMs more accessible and practical for real-world applications (p. 1, 10)."
    },
    {
        "pdf": "ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf",
        "question_type": "Question Combining Image and Text in PDF",
        "question": "How is the concept of prompt-chains explained and visually supported to demonstrate their role in improving model predictions?",
        "ground_truth": "Prompt-chains are described as reusable functional prompts that convert inputs into questions and then answers. The visual diagram shows how different chains yield varied predictions, which are then aggregated to improve accuracy (p. 2, Figure 1 and accompanying text)."
    },
    {
        "pdf": "ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf",
        "question_type": "Question Combining Image and Table in PDF",
        "question": "What relationship is shown between model size, aggregation method, and prediction uncertainty, and how is this supported by both visual and numerical data?",
        "ground_truth": "The paper shows that as model size increases and more prompts are aggregated, prediction uncertainty (measured by conditional entropy) decreases. This is supported by plots showing entropy trends and benchmark tables showing performance gains (p. 7, Figure 4; p. 8, Table 1)."
    },
    {
        "pdf": "ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf",
        "question_type": "Question Combining Table and Text in PDF",
        "question": "How do the authors justify the use of weak supervision over majority voting for aggregating prompt outputs, based on both empirical results and theoretical discussion?",
        "ground_truth": "Weak supervision accounts for prompt dependencies and varying accuracies, outperforming majority vote by up to 8.7 points. Theoretical analysis shows it reduces information loss during aggregation (p. 6–7, Section 3.4 and 4; p. 9, Table 5)."
    },
    {
        "pdf": "ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf",
        "question_type": "Question Combining Image, Text, and Table in PDF",
        "question": "How does the paper support the claim that AMA enables smaller models to match or exceed the performance of much larger models, using a combination of visual diagrams, benchmark results, and explanatory text?",
        "ground_truth": "The paper combines a visual explanation of prompt-chains (Figure 1), benchmark comparisons (Table 1), and detailed analysis of prompt effectiveness and aggregation strategies to show that GPT-J-6B with AMA outperforms GPT-3 175B on most tasks (p. 2, 4–8)."
    },

    # HC Overview Brochure PDF
    {
        "pdf": "HC Overview Brochure.pdf",
        "question_type": "Summarization of PDF",
        "question": "What is the main focus of the HC Overview Brochure?",
        "ground_truth": "The HC Overview Brochure focuses on Guidehouse's comprehensive suite of healthcare technology solutions, including cybersecurity, advanced analytics, enterprise resource planning, interoperability, and digital transformation (p. 2)."
    },
    {
        "pdf": "HC Overview Brochure.pdf",
        "question_type": "Question Related to Any Image in PDF",
        "question": "How does the brochure visually represent Guidehouse's Health IT capabilities?",
        "ground_truth": "The brochure includes a diagram that highlights Guidehouse's Health IT capabilities, such as cybersecurity, advanced analytics, enterprise resource planning, interoperability, and digital transformation (p. 2, Figure 1)."
    },
    {
        "pdf": "HC Overview Brochure.pdf",
        "question_type": "Question Related to Any Table in PDF",
        "question": "What information is provided in the table about Guidehouse's key contacts for Health IT solutions?",
        "ground_truth": "The table lists key contacts for various Health IT solutions, including their names, titles, and areas of expertise (p. 4, Table 1)."
    },
    {
        "pdf": "HC Overview Brochure.pdf",
        "question_type": "Question Related to Any Text in PDF",
        "question": "What is the significance of IT effectiveness and strategy in healthcare organizations, according to the brochure?",
        "ground_truth": "The brochure emphasizes that IT effectiveness and strategy are crucial for enhancing an organization's overall strategy, increasing value derived from existing IT investments, standardizing systems and applications, and optimizing operations to support the organization (p. 4)."
    },
    {
        "pdf": "HC Overview Brochure.pdf",
        "question_type": "Questions from Multiple Pages in PDF (2–3 Adjacent Pages)",
        "question": "How does the brochure describe the role of enterprise resource planning (ERP) in healthcare organizations?",
        "ground_truth": "The brochure explains that ERP is essential for creating efficiencies, digitally enabling corporate services, and aligning fragmented business processes. It also highlights Guidehouse's end-to-end solutions for ERP implementation and optimization (p. 5–6)."
    },
    {
        "pdf": "HC Overview Brochure.pdf",
        "question_type": "Questions from Start and End of PDF",
        "question": "How does the introduction frame the importance of Health IT solutions, and what conclusions are drawn about Guidehouse's expertise?",
        "ground_truth": "The introduction highlights the critical role of Health IT solutions in optimizing internal processes, empowering employees, and creating consumer-centric products. The conclusion emphasizes Guidehouse's extensive experience and expertise in providing comprehensive healthcare technology solutions (p. 1, 10)."
    },
    {
        "pdf": "HC Overview Brochure.pdf",
        "question_type": "Question Combining Image and Text in PDF",
        "question": "How does the brochure illustrate the impact of digital transformation on healthcare organizations?",
        "ground_truth": "The brochure includes both text and images to show how digital transformation enables healthcare organizations to innovate care delivery models, improve patient and provider experiences, and align reimbursement strategies (p. 7, Figure 2)."
    },
    {
        "pdf": "HC Overview Brochure.pdf",
        "question_type": "Question Combining Image and Table in PDF",
        "question": "What relationship is shown between Guidehouse's advanced analytics capabilities and their impact on healthcare organizations?",
        "ground_truth": "The brochure shows that Guidehouse's advanced analytics capabilities help healthcare organizations reduce costs, improve efficiencies, and enhance patient and employee satisfaction. This is supported by both visual diagrams and tables listing specific analytics solutions (p. 8, Figure 3; p. 9, Table 2)."
    },
    {
        "pdf": "HC Overview Brochure.pdf",
        "question_type": "Question Combining Table and Text in PDF",
        "question": "How does the brochure justify the importance of cybersecurity in healthcare organizations?",
        "ground_truth": "The brochure explains that cybersecurity is critical for protecting patient data and medical devices. It provides both empirical results and theoretical discussion on how Guidehouse's cybersecurity solutions help organizations become more resilient to cyber threats (p. 9, Section 4; p. 10, Table 3)."
    },
    {
        "pdf": "HC Overview Brochure.pdf",
        "question_type": "Question Combining Image, Text, and Table in PDF",
        "question": "How does the brochure support the claim that Guidehouse's Health IT solutions are comprehensive and effective, using a combination of visual diagrams, benchmark results, and explanatory text?",
        "ground_truth": "The brochure combines visual diagrams of Health IT capabilities (Figure 1), benchmark comparisons of solution effectiveness (Table 1), and detailed analysis of various Health IT solutions to demonstrate that Guidehouse's offerings are comprehensive and effective (p. 2, 4–10)."
    },

    # Artificial Intelligence PDF
    {
        "pdf": "Artificial Artificial Artificial Intelligence- Crowd Workers Widely Use Large Language Models for Text Production Tasks 1.pdf",
        "question_type": "Summarization of PDF",
        "question": "What is the main finding of the paper regarding the use of large language models (LLMs) by crowd workers?",
        "ground_truth": "The paper finds that a significant portion of crowd workers use LLMs to complete tasks, with an estimated 33–46% of summaries submitted by crowd workers being produced with the help of LLMs (p. 1)."
    },
    {
        "pdf": "Artificial Artificial Artificial Intelligence- Crowd Workers Widely Use Large Language Models for Text Production Tasks 1.pdf",
        "question_type": "Question Related to Any Image in PDF",
        "question": "How does the paper visually represent the methodology used to detect LLM-generated text?",
        "ground_truth": "The paper includes a diagram that illustrates the approach for quantifying the prevalence of LLM usage among crowd workers, showing the steps of training a synthetic-vs.-real classifier and validating results with keystroke data (p. 2, Figure 1)."
    },
    {
        "pdf": "Artificial Artificial Artificial Intelligence- Crowd Workers Widely Use Large Language Models for Text Production Tasks 1.pdf",
        "question_type": "Question Related to Any Table in PDF",
        "question": "What do the tables in the paper reveal about the performance of the synthetic-text detector?",
        "ground_truth": "The tables show that the synthetic-text detector achieves high accuracy and macro-F1 scores in both summary-level and abstract-level data splits, indicating its effectiveness in identifying LLM-generated text (p. 5, Table 1)."
    },
    {
        "pdf": "Artificial Artificial Artificial Intelligence- Crowd Workers Widely Use Large Language Models for Text Production Tasks 1.pdf",
        "question_type": "Question Related to Any Text in PDF",
        "question": "Why do the authors believe that the use of LLMs by crowd workers poses a challenge for crowdsourcing platforms?",
        "ground_truth": "The authors argue that the use of LLMs by crowd workers diminishes the utility of crowdsourced data, as it is no longer the intended human gold standard. This makes it difficult to assess the reliability and validity of the data (p. 1)."
    },
    {
        "pdf": "Artificial Artificial Artificial Intelligence- Crowd Workers Widely Use Large Language Models for Text Production Tasks 1.pdf",
        "question_type": "Questions from Multiple Pages in PDF (2–3 Adjacent Pages)",
        "question": "How does the paper develop the methodology for detecting LLM-generated text across the sections discussing task choice, synthetic text detection, and post-hoc validation?",
        "ground_truth": "The paper first describes the task of abstract summarization (Section 3.1), then details the process of training a synthetic-text detector (Section 3.2), and finally explains the post-hoc validation using keystroke data to confirm the results (Section 3.3, p. 3–5)."
    },
    {
        "pdf": "Artificial Artificial Artificial Intelligence- Crowd Workers Widely Use Large Language Models for Text Production Tasks 1.pdf",
        "question_type": "Questions from Start and End of PDF",
        "question": "How does the introduction frame the problem of LLM usage by crowd workers, and what conclusions are drawn about the broader implications?",
        "ground_truth": "The introduction highlights the concern that LLM usage by crowd workers undermines the reliability of crowdsourced data. The conclusion emphasizes the need for new methods to ensure that human data remains human, given the increasing prevalence of LLMs (p. 1, 7)."
    },
    {
        "pdf": "Artificial Artificial Artificial Intelligence- Crowd Workers Widely Use Large Language Models for Text Production Tasks 1.pdf",
        "question_type": "Question Combining Image and Text in PDF",
        "question": "How is the concept of synthetic-text detection explained and visually supported to demonstrate its effectiveness?",
        "ground_truth": "The paper describes the synthetic-text detection methodology and includes a visual diagram showing the steps of training a classifier and validating results with keystroke data, demonstrating the effectiveness of the approach (p. 2, Figure 1)."
    },
    {
        "pdf": "Artificial Artificial Artificial Intelligence- Crowd Workers Widely Use Large Language Models for Text Production Tasks 1.pdf",
        "question_type": "Question Combining Image and Table in PDF",
        "question": "What relationship is shown between the classifier's decision threshold and the proportion of summaries classified as synthetic, and how is this supported by both visual and numerical data?",
        "ground_truth": "The paper shows that the proportion of summaries classified as synthetic varies with the classifier's decision threshold. This is supported by a plot showing the relationship between the threshold and the proportion classified as synthetic, as well as tables showing classifier performance (p. 5, Figure 3; p. 6, Table 2)."
    },
    {
        "pdf": "Artificial Artificial Artificial Intelligence- Crowd Workers Widely Use Large Language Models for Text Production Tasks 1.pdf",
        "question_type": "Question Combining Table and Text in PDF",
        "question": "How do the authors justify the use of keystroke data for post-hoc validation of the synthetic-text detector, based on both empirical results and theoretical discussion?",
        "ground_truth": "The authors explain that keystroke data provides a high-precision method for distinguishing between human-generated and LLM-generated text. They provide empirical results showing low false-positive rates and theoretical discussion on the reliability of keystroke data (p. 5–6, Section 4; p. 6, Table 2)."
    },
    {
        "pdf": "Artificial Artificial Artificial Intelligence- Crowd Workers Widely Use Large Language Models for Text Production Tasks 1.pdf",
        "question_type": "Question Combining Image, Text, and Table in PDF",
        "question": "How does the paper support the claim that LLM usage by crowd workers is prevalent, using a combination of visual diagrams, classifier performance results, and explanatory text?",
        "ground_truth": "The paper combines a visual diagram of the detection methodology (Figure 1), classifier performance results (Table 1), and detailed analysis of keystroke data to show that a significant portion of crowd workers use LLMs to complete tasks (p. 2, 5–6)."
    },

    # ActionCLIP PDF
    {
        "pdf": "ActionCLIP- A New Paradigm for Video Action Recognition 1.pdf",
        "question_type": "Summarization of PDF",
        "question": "What is the main contribution of the ActionCLIP paper to the field of video action recognition?",
        "ground_truth": "The main contribution of the ActionCLIP paper is the introduction of a new paradigm for video action recognition that leverages multimodal learning with video and text, enabling zero-shot and few-shot action recognition without additional labeled data (p. 1)."
    },
    {
        "pdf": "ActionCLIP- A New Paradigm for Video Action Recognition 1.pdf",
        "question_type": "Question Related to Any Image in PDF",
        "question": "How does the paper visually represent the architecture of the ActionCLIP model?",
        "ground_truth": "The paper includes a diagram that illustrates the architecture of the ActionCLIP model, showing the components of the video encoder, text encoder, and similarity calculation module (p. 2, Figure 1)."
    },
    {
        "pdf": "ActionCLIP- A New Paradigm for Video Action Recognition 1.pdf",
        "question_type": "Question Related to Any Table in PDF",
        "question": "What do the tables in the paper reveal about the performance of ActionCLIP compared to other state-of-the-art methods?",
        "ground_truth": "The tables show that ActionCLIP achieves superior performance on several benchmark datasets, outperforming other state-of-the-art methods in terms of top-1 and top-5 accuracy (p. 6, Table 6)."
    },
    {
        "pdf": "ActionCLIP- A New Paradigm for Video Action Recognition 1.pdf",
        "question_type": "Question Related to Any Text in PDF",
        "question": "Why do the authors believe that multimodal learning is beneficial for video action recognition?",
        "ground_truth": "The authors argue that multimodal learning enhances the representation power of the model by incorporating semantic information from both video and text, enabling more accurate and flexible action recognition (p. 2)."
    },
    {
        "pdf": "ActionCLIP- A New Paradigm for Video Action Recognition 1.pdf",
        "question_type": "Questions from Multiple Pages in PDF (2–3 Adjacent Pages)",
        "question": "How does the paper develop the new paradigm for video action recognition across the sections discussing multimodal learning, pre-training, and fine-tuning?",
        "ground_truth": "The paper first introduces the multimodal learning framework (Section 3.1), then describes the pre-training process using large-scale web data (Section 3.2), and finally explains the fine-tuning process on target datasets (Section 3.3, p. 3–5)."
    },
    {
        "pdf": "ActionCLIP- A New Paradigm for Video Action Recognition 1.pdf",
        "question_type": "Questions from Start and End of PDF",
        "question": "How does the introduction frame the problem of traditional video action recognition methods, and what conclusions are drawn about the advantages of ActionCLIP?",
        "ground_truth": "The introduction highlights the limitations of traditional video action recognition methods, such as their reliance on predefined categories and labeled data. The conclusion emphasizes that ActionCLIP overcomes these limitations by leveraging multimodal learning and enabling zero-shot and few-shot recognition (p. 1, 7)."
    },
    {
        "pdf": "ActionCLIP- A New Paradigm for Video Action Recognition 1.pdf",
        "question_type": "Question Combining Image and Text in PDF",
        "question": "How is the concept of multimodal learning explained and visually supported to demonstrate its role in ActionCLIP?",
        "ground_truth": "The paper describes the multimodal learning framework and includes a visual diagram showing the integration of video and text encoders, as well as the similarity calculation module, to demonstrate the effectiveness of the approach (p. 2, Figure 1)."
    },
    {
        "pdf": "ActionCLIP- A New Paradigm for Video Action Recognition 1.pdf",
        "question_type": "Question Combining Image and Table in PDF",
        "question": "What relationship is shown between the number of input frames and the performance of ActionCLIP, and how is this supported by both visual and numerical data?",
        "ground_truth": "The paper shows that the performance of ActionCLIP improves with an increasing number of input frames. This is supported by a plot showing the relationship between input frames and accuracy, as well as tables showing performance results for different configurations (p. 6, Figure 3; p. 7, Table 7)."
    },
    {
        "pdf": "ActionCLIP- A New Paradigm for Video Action Recognition 1.pdf",
        "question_type": "Question Combining Table and Text in PDF",
        "question": "How do the authors justify the use of the new paradigm for video action recognition, based on both empirical results and theoretical discussion?",
        "ground_truth": "The authors explain that the new paradigm enables the model to leverage large-scale web data and adapt to new tasks without additional labeled data. They provide empirical results showing improved performance and theoretical discussion on the benefits of multimodal learning (p. 3–5, Section 3; p. 6, Table 6)."
    },
    {
        "pdf": "ActionCLIP- A New Paradigm for Video Action Recognition 1.pdf",
        "question_type": "Question Combining Image, Text, and Table in PDF",
        "question": "How does the paper support the claim that ActionCLIP is a cost-effective and efficient method for video action recognition, using a combination of visual diagrams, benchmark results, and explanatory text?",
        "ground_truth": "The paper combines a visual diagram of the ActionCLIP architecture (Figure 1), benchmark comparisons of performance (Table 6), and detailed analysis of runtime and computational efficiency to show that ActionCLIP is both cost-effective and efficient (p. 2, 6–7)."
    }
]
    
    # List to store results
    results = []
    
    # Process each question
    print(f"Processing {len(questions_answers)} questions...")
    for item in tqdm(questions_answers):
        pdf = item["pdf"]
        question_type = item["question_type"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        try:
            # Get model answer
            answer, sources = enhanced_process_query(question)
            
            # Calculate cosine similarity
            embedding1 = model.encode([ground_truth])
            embedding2 = model.encode([answer])
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            
            # Add to results
            results.append({
                "PDF": pdf,
                "Question Type": question_type,
                "Question": question,
                "Ground Truth Answer": ground_truth,
                "Model Answer": answer,
                "Sources": sources,
                "Cosine Similarity": similarity
            })
            
        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Error details: {str(e)}")
            traceback.print_exc()
            
            # Add to results with error message
            results.append({
                "PDF": pdf,
                "Question Type": question_type,
                "Question": question,
                "Ground Truth Answer": ground_truth,
                "Model Answer": f"ERROR: {str(e)}",
                "Sources": "",
                "Cosine Similarity": 0.0
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate average similarity (excluding errors)
    valid_df = df[df["Cosine Similarity"] > 0]
    
    if len(valid_df) > 0:
        avg_similarity = valid_df["Cosine Similarity"].mean()
        print(f"Average Cosine Similarity: {avg_similarity:.4f}")
        
        # Group by PDF and calculate average similarity per PDF
        pdf_avg = valid_df.groupby("PDF")["Cosine Similarity"].mean().reset_index()
        print("\nAverage Cosine Similarity by PDF:")
        for _, row in pdf_avg.iterrows():
            print(f"{row['PDF']}: {row['Cosine Similarity']:.4f}")
        
        # Group by Question Type and calculate average similarity
        type_avg = valid_df.groupby("Question Type")["Cosine Similarity"].mean().reset_index()
        print("\nAverage Cosine Similarity by Question Type:")
        for _, row in type_avg.iterrows():
            print(f"{row['Question Type']}: {row['Cosine Similarity']:.4f}")
    else:
        print("No valid similarities calculated")
    
    # Save to Excel
    print(f"Saving results to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
        
        if len(valid_df) > 0:
            # Create summary sheet
            summary_data = {
                "Metric": ["Overall Average", "Questions Processed", "Successful Queries", "Failed Queries"],
                "Value": [
                    avg_similarity,
                    len(df),
                    len(valid_df),
                    len(df) - len(valid_df)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # PDF averages
            pdf_avg.to_excel(writer, sheet_name="PDF Averages", index=False)
            
            # Question type averages
            type_avg.to_excel(writer, sheet_name="Question Type Averages", index=False)
    
    print(f"Evaluation complete! Results saved to {output_file}")
    return output_file

if __name__ == "__main__":
    evaluate_rag_system()