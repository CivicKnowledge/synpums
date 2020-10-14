# Copyright (c) 2020 Civic Knowledge. This file is licensed under the terms of the
# MIT license included in this distribution as LICENSE

from enum import Enum, auto


class FamilyType(Enum):
    FAMILY_MARRIED = auto()
    FAMILY_OTHER = auto()
    FAMILY_MALE_NW = auto()
    FAMILY_FEMALE_NH = auto()
    NONFAMILY_LIVE_ALONE = auto()
    NONFAMILY_NOTLIVE_ALONE = auto()

class Sex(Enum):
    MALE = auto()
    FEMALE = auto()

# No schooling completed
# Nursery to 4th grade
# 5th and 6th grade
# 7th and 8th grade
# 9th grade
# 10th grade
# 11th grade
# 12th grade no diploma
# High school graduate (includes equivalency)
# Some college less than 1 year
# Some college 1 or more years no degree
# Associate's degree
# Bachelor's degree
# Master's degree
# Professional school degree
# Doctorate degree
class EducationalAttainment(Enum):
    KG4 = auto()
    G5_6 = auto()
    G7_8 = auto()
    G9 = auto()
    G10 = auto()
    G11 = auto()
    G12 = auto()
    HS = auto()
    SOME_COLLEGE_1Y = auto()
    SOME_COLLEGE = auto()
    ASSOC = auto()
    BACHELORS = auto()
    MASTERS = auto()
    PROFESSIONAL = auto()
    DOCTORATE = auto()

# Table B14001: School Enrollment by Level of School
# B14001_001  Population 3 years and over  - Total
# B14001_002 Population 3 years and over - Enrolled in school
# B14001_003 Population 3 years and over - Enrolled in school  - Enrolled in nursery school preschool
# B14001_004 Population 3 years and over - Enrolled in school  - Enrolled in kindergarten
# B14001_005 Population 3 years and over - Enrolled in school  - Enrolled in grade 1 to grade 4
# B14001_006 Population 3 years and over - Enrolled in school  - Enrolled in grade 5 to grade 8
# B14001_007 Population 3 years and over - Enrolled in school  - Enrolled in grade 9 to grade 12
# B14001_008 Population 3 years and over - Enrolled in school  - Enrolled in college undergraduate years
# B14001_009 Population 3 years and over - Enrolled in school  - Graduate or professional school
# B14001_010 Population 3 years and over - Not enrolled in school

class SchoolEnrollment(Enum):
    PREK = auto()
    K = auto()
    G1_4 = auto()
    G5_8 = auto()
    G9_12 = auto()
    GCOLLEGE = auto()
    GGRADUATE = auto()
