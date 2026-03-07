# config.py
from pathlib import Path

CKAN_BASE = "https://open.canada.ca/data/en/api/3/action"
CKAN_PACKAGE_SEARCH_URL = f"{CKAN_BASE}/package_search"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DB_PATH = Path("data/db/ground_truth.sqlite")

USER_AGENT = "AiEO-Research/1.0 (+student research project)"

SEARCH_QUERIES = [
    "Indigenous skills employment training",
    "youth employment program",
    "newcomer employment settlement",
    "disability employment support",
    "apprenticeship training canada",
    "mental health services canada",
    "disability support services",
    "rural remote health services",
    "indigenous health program",
    "community grants funding",
    "youth funding grant",
    "indigenous community fund",
    "newcomer settlement fund",
    "post-secondary education indigenous",
    "literacy training adult",
    "language training newcomer",
]

ROWS_PER_QUERY = 100

REGION_VALUES = [
    "national", "BC", "AB", "SK", "MB", "ON",
    "QC", "NB", "NS", "PEI", "NL", "YT", "NT", "NU"
]

POPULATION_VALUES = [
    "Indigenous", "First_Nations", "Metis", "Inuit",
    "newcomers", "youth", "disabilities",
    "rural_remote", "women", "seniors", "general_public"
]

SECTOR_VALUES = [
    "employment", "skills_training", "education",
    "healthcare", "mental_health", "housing",
    "funding_grants", "public_services",
    "immigration_settlement", "language_training",
    "disability_support", "digital_access"
]

POPULATION_KEYWORDS = {
    "Indigenous": ["indigenous", "first nation", "métis", "metis", "inuit", "aboriginal", "fnmi"],
    "First_Nations": ["first nation", "band council", "treaty"],
    "Metis": ["métis", "metis"],
    "Inuit": ["inuit", "inuk", "nunavut", "inuvialuit"],
    "newcomers": ["newcomer", "immigrant", "refugee", "settlement", "ircc", "resettlement"],
    "youth": ["youth", "young", "student", "apprentice", "15-30", "under 30"],
    "disabilities": ["disability", "disabilities", "accessible", "accessibility", "aoda"],
    "rural_remote": ["rural", "remote", "northern", "northern communities", "fly-in"],
    "women": ["women", "woman", "gender", "femme"],
    "seniors": ["senior", "elder", "aging", "older adult"],
}

SECTOR_KEYWORDS = {
    "employment": ["employment", "job", "work", "labour", "labor", "workforce"],
    "skills_training": ["training", "apprenticeship", "skills", "upskilling", "reskilling"],
    "education": ["education", "school", "post-secondary", "university", "college", "literacy"],
    "healthcare": ["health", "medical", "clinic", "hospital", "wellness"],
    "mental_health": ["mental health", "counselling", "counseling", "psychology", "wellbeing"],
    "housing": ["housing", "shelter", "homelessness", "rent"],
    "funding_grants": ["grant", "fund", "funding", "contribution", "bursary", "scholarship"],
    "immigration_settlement": ["immigration", "settlement", "newcomer", "refugee", "ircc"],
    "language_training": ["language", "english", "french", "esl", "fsl", "ells"],
    "disability_support": ["disability", "accessible", "accommodation", "aoda", "barrier-free"],
}