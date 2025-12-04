#!/usr/bin/env python3
"""
Smart Study Assistant - Terminal Version
An improved educational tool that finds YouTube videos and generates quiz questions.
"""

import os
import re
import sys
import math
import time
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from colorama import init, Fore, Style
from groq import Groq
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()
init(autoreset=True)


class Config:
    """Configuration management for API keys and settings."""
    
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
    
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    MAX_SEARCH_RESULTS: int = 10
    MIN_VIDEO_DURATION: int = 240
    MAX_VIDEO_DURATION: int = 720
    NUM_QUESTIONS: int = 6
    
    EDUCATIONAL_KEYWORDS: List[str] = [
        "explanation", "concept", "chapter", "lecture", 
        "introduction", "easy", "tutorial", "learn", "basics"
    ]
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are present."""
        if not cls.GROQ_API_KEY:
            print_error("Missing required API key: GROQ_API_KEY")
            print_info("Please set it in your .env file or environment variables.")
            print_info("Get your free key at: https://console.groq.com")
            return False
        return True
    
    @classmethod
    def has_youtube_key(cls) -> bool:
        """Check if YouTube API key is available."""
        return bool(cls.YOUTUBE_API_KEY)


def print_header(text: str) -> None:
    """Print a styled header."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*50}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(50)}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*50}{Style.RESET_ALL}\n")


def print_success(text: str) -> None:
    """Print success message in green."""
    print(f"{Fore.GREEN}{Style.BRIGHT}[SUCCESS]{Style.RESET_ALL} {text}")


def print_error(text: str) -> None:
    """Print error message in red."""
    print(f"{Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL} {text}")


def print_warning(text: str) -> None:
    """Print warning message in yellow."""
    print(f"{Fore.YELLOW}{Style.BRIGHT}[WARNING]{Style.RESET_ALL} {text}")


def print_info(text: str) -> None:
    """Print info message in blue."""
    print(f"{Fore.BLUE}{Style.BRIGHT}[INFO]{Style.RESET_ALL} {text}")


def print_progress(text: str) -> None:
    """Print progress indicator."""
    print(f"{Fore.MAGENTA}>>> {text}...{Style.RESET_ALL}")


def get_valid_input(prompt: str, min_length: int = 1) -> str:
    """Get validated user input with minimum length requirement."""
    while True:
        user_input = input(f"{Fore.YELLOW}{prompt}{Style.RESET_ALL}").strip()
        if len(user_input) >= min_length:
            return user_input
        print_warning(f"Please enter at least {min_length} character(s).")


def parse_duration_to_seconds(duration: str) -> Optional[int]:
    """
    Convert ISO 8601 duration format to seconds.
    Example: PT1H30M45S -> 5445 seconds
    """
    if not duration:
        return None
    
    match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return None
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: int) -> str:
    """Format seconds into human-readable duration."""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def get_grade_level_info(standard: str) -> Dict[str, Any]:
    """Get grade level information for better video matching."""
    std_lower = standard.lower().strip()
    
    try:
        grade_num = int(re.search(r'\d+', std_lower).group()) if re.search(r'\d+', std_lower) else 0
    except:
        grade_num = 0
    
    if grade_num >= 11 or any(x in std_lower for x in ['college', 'university', '11', '12', 'jee', 'neet']):
        return {
            "level": "advanced",
            "keywords": ["jee", "neet", "ncert", "class 11", "class 12", "advanced", "detailed", "in-depth", "cbse", "board"],
            "avoid": ["kids", "children", "basic", "nursery", "kindergarten", "primary", "elementary", "class 1", "class 2", "class 3", "class 4", "class 5"],
            "grade": grade_num
        }
    elif grade_num >= 9 or any(x in std_lower for x in ['9', '10', 'high school', 'secondary']):
        return {
            "level": "intermediate",
            "keywords": ["class 9", "class 10", "ncert", "cbse", "icse", "high school", "secondary", "board exam", "detailed"],
            "avoid": ["kids", "children", "for kids", "nursery", "kindergarten", "primary", "elementary", "class 1", "class 2", "class 3", "class 4", "class 5", "class 6"],
            "grade": grade_num
        }
    elif grade_num >= 6:
        return {
            "level": "middle",
            "keywords": ["class 6", "class 7", "class 8", "middle school", "ncert"],
            "avoid": ["kids", "toddler", "nursery", "kindergarten", "class 1", "class 2", "class 3"],
            "grade": grade_num
        }
    else:
        return {
            "level": "beginner",
            "keywords": ["easy", "simple", "basics", "introduction", "beginner"],
            "avoid": ["advanced", "jee", "neet", "competitive"],
            "grade": grade_num
        }


def calculate_video_score(
    title: str, 
    description: str, 
    duration_secs: int, 
    view_count: int,
    topic: str,
    grade_info: Dict[str, Any]
) -> float:
    """
    Calculate a relevance score for a video based on multiple factors.
    Higher score = more suitable for educational purposes and grade level.
    """
    score = 0.0
    title_lower = title.lower()
    desc_lower = description.lower()
    topic_lower = topic.lower()
    combined_text = title_lower + " " + desc_lower
    
    if Config.MIN_VIDEO_DURATION <= duration_secs <= Config.MAX_VIDEO_DURATION:
        score += 5.0
    elif duration_secs < Config.MIN_VIDEO_DURATION:
        score += 2.0
    else:
        score += 1.0
    
    for keyword in Config.EDUCATIONAL_KEYWORDS:
        if keyword in title_lower:
            score += 2.0
        if keyword in desc_lower:
            score += 1.0
    
    topic_words = topic_lower.split()
    for word in topic_words:
        if len(word) > 2 and word in title_lower:
            score += 3.0
        if len(word) > 2 and word in desc_lower:
            score += 1.5
    
    if view_count > 0:
        score += min(4.0, math.log10(view_count + 1))
    
    for keyword in grade_info.get("keywords", []):
        if keyword in combined_text:
            score += 4.0
    
    for avoid_word in grade_info.get("avoid", []):
        if avoid_word in combined_text:
            score -= 8.0
    
    grade = grade_info.get("grade", 0)
    if grade > 0:
        grade_pattern = rf'class\s*{grade}|grade\s*{grade}|std\s*{grade}'
        if re.search(grade_pattern, combined_text):
            score += 6.0
    
    quality_channels = ["vedantu", "byju", "unacademy", "khan academy", "physics wallah", "ncert"]
    for channel in quality_channels:
        if channel in combined_text:
            score += 3.0
    
    return round(score, 2)


class YouTubeSearcher:
    """Handles YouTube video search and scoring."""
    
    def __init__(self, api_key: str):
        self.youtube = build("youtube", "v3", developerKey=api_key)
    
    def search_best_video(self, topic: str, standard: str) -> Optional[Dict[str, Any]]:
        """
        Search for the best educational video on a topic.
        Returns video details including URL, title, and score.
        """
        grade_info = get_grade_level_info(standard)
        
        if grade_info["level"] == "advanced":
            query = f"{topic} class {standard} NCERT detailed explanation"
        elif grade_info["level"] == "intermediate":
            query = f"{topic} class {standard} NCERT CBSE explanation"
        else:
            query = f"{topic} class {standard} explanation tutorial"
        
        try:
            print_progress(f"Searching YouTube for {grade_info['level']}-level videos")
            
            search_response = self.youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=Config.MAX_SEARCH_RESULTS * 2,
                order="relevance",
                videoDuration="medium",
                safeSearch="strict"
            ).execute()
            
            video_ids = [
                item["id"]["videoId"] 
                for item in search_response.get("items", [])
            ]
            
            if not video_ids:
                return None
            
            print_progress("Analyzing video quality and grade-level match")
            
            details_response = self.youtube.videos().list(
                part="contentDetails,snippet,statistics",
                id=",".join(video_ids)
            ).execute()
            
            candidates = []
            
            for item in details_response.get("items", []):
                video_id = item["id"]
                title = item["snippet"]["title"]
                description = item["snippet"].get("description", "")
                channel = item["snippet"].get("channelTitle", "Unknown")
                
                duration_iso = item["contentDetails"].get("duration")
                duration_secs = parse_duration_to_seconds(duration_iso)
                view_count = int(item["statistics"].get("viewCount", 0))
                
                if duration_secs is None:
                    continue
                
                score = calculate_video_score(
                    title, description, duration_secs, view_count, topic, grade_info
                )
                
                candidates.append({
                    "id": video_id,
                    "title": title,
                    "description": description[:500],
                    "channel": channel,
                    "duration": duration_secs,
                    "views": view_count,
                    "score": score,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                })
            
            if not candidates:
                return None
            
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[0]
            
        except HttpError as e:
            print_error(f"YouTube API error: {e.resp.status} - {e.content.decode()}")
            return None
        except Exception as e:
            print_error(f"Error searching YouTube: {str(e)}")
            return None


class QuizGenerator:
    """Handles question generation and answer evaluation using Groq."""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = Config.GROQ_MODEL
    
    def generate_questions(self, topic: str, video_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate 4 MCQs and 2 short-answer questions based on video content."""
        
        if video_info:
            video_context = f"""
VIDEO INFORMATION:
Title: {video_info.get('title', '')}
Channel: {video_info.get('channel', '')}
Description: {video_info.get('description', '')}

Generate questions that a student would be able to answer AFTER watching this specific video.
Focus on concepts that would likely be covered in this video based on its title and description."""
        else:
            video_context = "Generate general questions about this topic."
        
        prompt = f"""Generate a quiz about the topic: "{topic}".

{video_context}

Create exactly 4 multiple choice questions (MCQ) and 2 simple short-answer questions.

RULES:
- Questions MUST be based on what would be taught in the video
- Keep questions simple and appropriate for the topic
- MCQ options should be realistic and related to the topic
- Short answers should need only 1-3 words

FORMAT:
MCQ QUESTIONS (4):
1. [Question based on video content]
   A) [Option A]
   B) [Option B]
   C) [Option C]
   D) [Option D]
   Answer: [Correct letter]

2. [Question based on video content]
   A) [Option A]
   B) [Option B]
   C) [Option C]
   D) [Option D]
   Answer: [Correct letter]

3. [Question based on video content]
   A) [Option A]
   B) [Option B]
   C) [Option C]
   D) [Option D]
   Answer: [Correct letter]

4. [Question based on video content]
   A) [Option A]
   B) [Option B]
   C) [Option C]
   D) [Option D]
   Answer: [Correct letter]

SHORT ANSWER QUESTIONS (2):
5. [Simple question requiring 1-2 word answer]
   Answer: [Correct answer]

6. [Simple question requiring 1-2 word answer]
   Answer: [Correct answer]"""

        try:
            print_progress("Generating quiz questions")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content or ""
            
            return self._parse_quiz(content)
            
        except Exception as e:
            print_error(f"Error generating questions: {str(e)}")
            return {"mcq": [], "short": [], "raw": ""}
    
    def _parse_quiz(self, content: str) -> Dict[str, Any]:
        """Parse the generated quiz content into structured format."""
        mcq_questions = []
        short_questions = []
        
        lines = content.split("\n")
        current_question = None
        current_options = []
        current_answer = ""
        in_mcq = True
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "SHORT ANSWER" in line.upper():
                in_mcq = False
                continue
            
            if re.match(r'^[1-4][\.\)]', line):
                if current_question and current_options:
                    mcq_questions.append({
                        "question": current_question,
                        "options": current_options.copy(),
                        "answer": current_answer
                    })
                current_question = re.sub(r'^[1-4][\.\)]\s*', '', line)
                current_options = []
                current_answer = ""
            
            elif re.match(r'^[5-6][\.\)]', line):
                if current_question and in_mcq and current_options:
                    mcq_questions.append({
                        "question": current_question,
                        "options": current_options.copy(),
                        "answer": current_answer
                    })
                current_question = re.sub(r'^[5-6][\.\)]\s*', '', line)
                current_options = []
                current_answer = ""
                in_mcq = False
            
            elif re.match(r'^[A-D][\)\.]', line):
                option = line
                current_options.append(option)
            
            elif line.lower().startswith("answer:"):
                current_answer = line.split(":", 1)[1].strip()
                if not in_mcq and current_question:
                    short_questions.append({
                        "question": current_question,
                        "answer": current_answer
                    })
                    current_question = None
        
        if current_question:
            if in_mcq and current_options:
                mcq_questions.append({
                    "question": current_question,
                    "options": current_options.copy(),
                    "answer": current_answer
                })
            elif not in_mcq:
                short_questions.append({
                    "question": current_question,
                    "answer": current_answer
                })
        
        return {
            "mcq": mcq_questions[:4],
            "short": short_questions[:2],
            "raw": content
        }
    
    def evaluate_answers(
        self, 
        topic: str, 
        questions: List[str], 
        answers: List[str]
    ) -> Dict[str, Any]:
        """Evaluate user answers and provide detailed feedback."""
        
        qa_pairs = "\n".join([
            f"{q}\nStudent Answer: {a}" 
            for q, a in zip(questions, answers)
        ])
        
        prompt = f"""You are evaluating a student's answers about "{topic}".

For each question and answer pair below, evaluate if the answer is correct, partially correct, or incorrect.

{qa_pairs}

Provide your evaluation in this exact format:
EVALUATION:
Q1: [CORRECT/PARTIAL/INCORRECT] - [Brief explanation]
Q2: [CORRECT/PARTIAL/INCORRECT] - [Brief explanation]
(continue for all questions...)

SCORE: X/{len(questions)}
(Count CORRECT as 1 point, PARTIAL as 0.5 points, INCORRECT as 0 points)

FEEDBACK:
[2-3 sentences of encouraging feedback and suggestions for improvement]"""

        try:
            print_progress("Evaluating your answers")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content or ""
            
            result_lower = result.lower()
            correct = result_lower.count("correct") - result_lower.count("incorrect")
            partial = result_lower.count("partial")
            
            return {
                "evaluation": result,
                "estimated_correct": max(0, correct),
                "partial": partial
            }
            
        except Exception as e:
            print_error(f"Error evaluating answers: {str(e)}")
            return {"evaluation": "Could not evaluate answers.", "estimated_correct": 0}


def display_video_info(video: Dict[str, Any]) -> None:
    """Display video information in a formatted way."""
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Best Video Found!{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'─'*50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Title:{Style.RESET_ALL} {video['title']}")
    print(f"{Fore.CYAN}Channel:{Style.RESET_ALL} {video['channel']}")
    print(f"{Fore.CYAN}Duration:{Style.RESET_ALL} {format_duration(video['duration'])}")
    print(f"{Fore.CYAN}Views:{Style.RESET_ALL} {video['views']:,}")
    print(f"{Fore.CYAN}Relevance Score:{Style.RESET_ALL} {video['score']}/20")
    print(f"{Fore.WHITE}{'─'*50}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}Watch here:{Style.RESET_ALL} {video['url']}")


def display_and_collect_quiz(quiz: Dict[str, Any]) -> Dict[str, Any]:
    """Display quiz questions and collect answers."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*50}")
    print(f"{'QUIZ TIME!'.center(50)}")
    print(f"{'='*50}{Style.RESET_ALL}")
    
    mcq_answers = []
    short_answers = []
    correct_answers = []
    
    if quiz.get("mcq"):
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}MULTIPLE CHOICE QUESTIONS (4){Style.RESET_ALL}")
        print(f"{Fore.WHITE}Enter A, B, C, or D for each question{Style.RESET_ALL}\n")
        
        for i, mcq in enumerate(quiz["mcq"], 1):
            print(f"{Fore.CYAN}{Style.BRIGHT}Q{i}.{Style.RESET_ALL} {mcq['question']}")
            for option in mcq.get("options", []):
                print(f"   {Fore.WHITE}{option}{Style.RESET_ALL}")
            
            while True:
                answer = input(f"{Fore.GREEN}Your Answer (A/B/C/D): {Style.RESET_ALL}").strip().upper()
                if answer in ['A', 'B', 'C', 'D']:
                    mcq_answers.append(answer)
                    correct_answers.append(mcq.get("answer", "").upper().strip())
                    break
                print_warning("Please enter A, B, C, or D")
            print()
    
    if quiz.get("short"):
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}SHORT ANSWER QUESTIONS (2){Style.RESET_ALL}")
        print(f"{Fore.WHITE}Type a brief answer (1-3 words){Style.RESET_ALL}\n")
        
        for i, short in enumerate(quiz["short"], 5):
            print(f"{Fore.CYAN}{Style.BRIGHT}Q{i}.{Style.RESET_ALL} {short['question']}")
            answer = input(f"{Fore.GREEN}Your Answer: {Style.RESET_ALL}").strip()
            short_answers.append(answer if answer else "[No answer]")
            correct_answers.append(short.get("answer", ""))
            print()
    
    return {
        "mcq_answers": mcq_answers,
        "short_answers": short_answers,
        "correct_answers": correct_answers,
        "quiz": quiz
    }


def evaluate_quiz(mcq_answers: List[str], short_answers: List[str], quiz: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the quiz answers and calculate score."""
    total = 0
    correct = 0
    results = []
    
    for i, (user_ans, mcq) in enumerate(zip(mcq_answers, quiz.get("mcq", [])), 1):
        correct_ans = mcq.get("answer", "").upper().strip()
        if len(correct_ans) > 1:
            correct_ans = correct_ans[0]
        is_correct = user_ans == correct_ans
        if is_correct:
            correct += 1
        total += 1
        results.append({
            "q_num": i,
            "type": "MCQ",
            "user_answer": user_ans,
            "correct_answer": correct_ans,
            "is_correct": is_correct
        })
    
    for i, (user_ans, short) in enumerate(zip(short_answers, quiz.get("short", [])), 5):
        correct_ans = short.get("answer", "").lower().strip()
        is_correct = user_ans.lower().strip() == correct_ans or correct_ans in user_ans.lower()
        if is_correct:
            correct += 1
        total += 1
        results.append({
            "q_num": i,
            "type": "Short",
            "user_answer": user_ans,
            "correct_answer": short.get("answer", ""),
            "is_correct": is_correct
        })
    
    return {
        "results": results,
        "correct": correct,
        "total": total,
        "percentage": round((correct / total) * 100) if total > 0 else 0
    }


def display_quiz_results(evaluation: Dict[str, Any]) -> None:
    """Display quiz results in a formatted way."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*50}")
    print(f"{'QUIZ RESULTS'.center(50)}")
    print(f"{'='*50}{Style.RESET_ALL}\n")
    
    for result in evaluation.get("results", []):
        q_num = result["q_num"]
        q_type = result["type"]
        user_ans = result["user_answer"]
        correct_ans = result["correct_answer"]
        is_correct = result["is_correct"]
        
        if is_correct:
            status = f"{Fore.GREEN}CORRECT{Style.RESET_ALL}"
        else:
            status = f"{Fore.RED}INCORRECT{Style.RESET_ALL}"
        
        print(f"{Fore.CYAN}Q{q_num} ({q_type}):{Style.RESET_ALL} {status}")
        print(f"   Your answer: {user_ans}")
        if not is_correct:
            print(f"   Correct answer: {Fore.YELLOW}{correct_ans}{Style.RESET_ALL}")
        print()
    
    correct = evaluation.get("correct", 0)
    total = evaluation.get("total", 6)
    percentage = evaluation.get("percentage", 0)
    
    print(f"{Fore.WHITE}{'─'*50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}FINAL SCORE: {correct}/{total} ({percentage}%){Style.RESET_ALL}")
    
    if percentage >= 80:
        print(f"{Fore.GREEN}Excellent work! You've mastered this topic!{Style.RESET_ALL}")
    elif percentage >= 60:
        print(f"{Fore.YELLOW}Good job! Review the incorrect answers to improve.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Keep practicing! Watch the video again and try once more.{Style.RESET_ALL}")


def display_results(result: Dict[str, Any]) -> None:
    """Display evaluation results."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*50}")
    print(f"{'QUIZ RESULTS'.center(50)}")
    print(f"{'='*50}{Style.RESET_ALL}\n")
    
    print(result["evaluation"])


def run_study_session() -> None:
    """Main study session flow."""
    print_header("SMART STUDY ASSISTANT")
    print(f"{Fore.WHITE}Your personal learning companion for video lessons and quizzes!{Style.RESET_ALL}\n")
    
    if not Config.validate():
        sys.exit(1)
    
    topic = get_valid_input("Enter the topic you want to study: ", min_length=2)
    standard = get_valid_input("Enter your class/grade level: ", min_length=1)
    
    print()
    
    video = None
    if Config.has_youtube_key():
        youtube_searcher = YouTubeSearcher(Config.YOUTUBE_API_KEY)
        video = youtube_searcher.search_best_video(topic, standard)
        
        if video:
            display_video_info(video)
            print(f"\n{Fore.YELLOW}Tip: Watch the video before attempting the quiz!{Style.RESET_ALL}")
            input(f"\n{Fore.WHITE}Press Enter when you're ready for the quiz...{Style.RESET_ALL}")
        else:
            print_warning("No suitable video found for this topic.")
            print_info("Continuing with quiz questions...\n")
    else:
        print_info("YouTube video search is disabled (no API key provided).")
        print_info("Proceeding directly to quiz questions...\n")
    
    quiz_generator = QuizGenerator(Config.GROQ_API_KEY)
    quiz = quiz_generator.generate_questions(topic, video)
    
    if not quiz.get("mcq") and not quiz.get("short"):
        print_error("Could not generate questions. Please try again.")
        return
    
    user_responses = display_and_collect_quiz(quiz)
    
    evaluation = evaluate_quiz(
        user_responses["mcq_answers"],
        user_responses["short_answers"],
        quiz
    )
    
    display_quiz_results(evaluation)
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*50}")
    print(f"{'SESSION COMPLETE'.center(50)}")
    print(f"{'='*50}{Style.RESET_ALL}")
    print(f"\n{Fore.GREEN}Great job completing this study session!{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Topic studied: {topic}{Style.RESET_ALL}")
    if video:
        print(f"{Fore.WHITE}Video watched: {video['title'][:40]}...{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Keep learning and improving!{Style.RESET_ALL}\n")


def main():
    """Entry point with error handling."""
    try:
        run_study_session()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Session interrupted. Goodbye!{Style.RESET_ALL}\n")
        sys.exit(0)
    except Exception as e:
        print_error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
