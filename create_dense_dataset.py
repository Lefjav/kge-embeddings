#!/usr/bin/env python3
"""
Create a simple MITRE dataset with symmetric relationships like synthetic data
"""

import xml.etree.ElementTree as ET
import os
import random
from collections import defaultdict

def create_simple_mitre_dataset():
    """
    Create a simple dataset with symmetric relationships.
    """
    print("Creating simple MITRE dataset...")
    
    # Parse hierarchical relationships
    capec_triples = parse_capec_hierarchical_only("mitre-data/raw/capec_v3.9.xml")
    cwe_triples = parse_cwe_hierarchical_only("mitre-data/raw/cwec_v4.17.xml")
    
    # Parse CWE-CAPEC attack pattern relationships from both directions
    cwe_capec_triples = parse_cwe_capec_relationships("mitre-data/raw/cwec_v4.17.xml")
    capec_cwe_triples = parse_capec_cwe_relationships("mitre-data/raw/capec_v3.9.xml")
    
    # Combine both directions
    all_cwe_capec_triples = cwe_capec_triples + capec_cwe_triples
    
    all_triples = capec_triples + cwe_triples + all_cwe_capec_triples
    print(f"Total hierarchical triples: {len(capec_triples + cwe_triples)}")
    print(f"Total CWE-CAPEC triples (CWE→CAPEC): {len(cwe_capec_triples)}")
    print(f"Total CWE-CAPEC triples (CAPEC→CWE): {len(capec_cwe_triples)}")
    print(f"Total CWE-CAPEC triples (both directions): {len(all_cwe_capec_triples)}")
    print(f"Total all triples: {len(all_triples)}")
    
    # Count entity frequencies
    entity_counts = defaultdict(int)
    for head, relation, tail in all_triples:
        entity_counts[head] += 1
        entity_counts[tail] += 1
    
    # Find entities that appear at least 4 times
    frequent_entities = set()
    for entity, count in entity_counts.items():
        if count >= 4:
            frequent_entities.add(entity)
    
    print(f"Found {len(frequent_entities)} entities appearing 4+ times")
    
    # Create simple symmetric relationships
    simple_triples = []
    entity_list = list(frequent_entities)
    
    # Create relationships between entities that are connected in the original data
    connected_pairs = set()
    for head, relation, tail in all_triples:
        if head in frequent_entities and tail in frequent_entities:
            connected_pairs.add((head, tail))
            connected_pairs.add((tail, head))  # Make it symmetric
    
    # Convert to simple relationships
    for entity1, entity2 in connected_pairs:
        if entity1 != entity2:
            # Create simple symmetric relationship
            simple_triples.append((entity1, "related_to", entity2))
    
    print(f"Simple triples: {len(simple_triples)}")
    
    # Split data
    random.shuffle(simple_triples)
    train_count = int(len(simple_triples) * 0.7)
    valid_count = int(len(simple_triples) * 0.15)
    
    train_triples = simple_triples[:train_count]
    valid_triples = simple_triples[train_count:train_count + valid_count]
    test_triples = simple_triples[train_count + valid_count:]
    
    # Write files
    output_dir = "mitre_data_dense"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "train.txt"), 'w') as f:
        for head, relation, tail in train_triples:
            f.write(f"{head}\t{relation}\t{tail}\n")
    
    with open(os.path.join(output_dir, "valid.txt"), 'w') as f:
        for head, relation, tail in valid_triples:
            f.write(f"{head}\t{relation}\t{tail}\n")
    
    with open(os.path.join(output_dir, "test.txt"), 'w') as f:
        for head, relation, tail in test_triples:
            f.write(f"{head}\t{relation}\t{tail}\n")
    
    print(f"Simple dataset created in {output_dir}/")
    print(f"Train: {len(train_triples)}, Valid: {len(valid_triples)}, Test: {len(test_triples)}")
    
    # Show sample
    print("\nSample triples:")
    for i, triple in enumerate(train_triples[:10]):
        print(f"  {i+1}: {triple[0]} -> {triple[1]} -> {triple[2]}")
    
    # Show CWE-CAPEC relationships specifically
    print("\nCWE-CAPEC relationships found:")
    cwe_capec_count = 0
    for triple in train_triples:
        if triple[0].startswith('CWE-') and triple[1] == 'related_to' and triple[2].startswith('CAPEC-'):
            print(f"  {triple[0]} -> {triple[1]} -> {triple[2]}")
            cwe_capec_count += 1
            if cwe_capec_count >= 10:  # Show first 10
                break
    
    print(f"Total CWE-CAPEC relationships in training: {cwe_capec_count}")
    
    # Specifically look for CWE-79
    print("\nCWE-79 specific relationships:")
    cwe79_found = False
    for triple in train_triples + valid_triples + test_triples:
        if "CWE-79" in triple[0] or "CWE-79" in triple[2]:
            print(f"  {triple[0]} -> {triple[1]} -> {triple[2]}")
            cwe79_found = True
    
    if not cwe79_found:
        print("  CWE-79 NOT FOUND in any relationships!")
    
    return train_triples, valid_triples, test_triples

def parse_capec_hierarchical_only(xml_path: str):
    """Parse CAPEC hierarchical relationships only."""
    triples = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {'capec': 'http://capec.mitre.org/capec-3'}
        
        for attack_pattern in root.findall('.//capec:Attack_Pattern', ns):
            pattern_id = attack_pattern.get('ID')
            if not pattern_id:
                continue
                
            capec_entity = f"CAPEC-{pattern_id}"
            
            related_patterns = attack_pattern.find('.//capec:Related_Attack_Patterns', ns)
            if related_patterns is not None:
                for related in related_patterns.findall('capec:Related_Attack_Pattern', ns):
                    related_id = related.get('CAPEC_ID')
                    nature = related.get('Nature')
                    
                    if related_id and nature in ['ChildOf', 'ParentOf']:
                        related_entity = f"CAPEC-{related_id}"
                        
                        if nature == "ChildOf":
                            triples.append((capec_entity, "is_child_of", related_entity))
                            triples.append((related_entity, "has_child", capec_entity))
                        elif nature == "ParentOf":
                            triples.append((capec_entity, "is_parent_of", related_entity))
                            triples.append((related_entity, "has_parent", capec_entity))
    except Exception as e:
        print(f"Error parsing CAPEC: {e}")
    
    return triples

def parse_cwe_hierarchical_only(xml_path: str):
    """Parse CWE hierarchical relationships only."""
    triples = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}
        
        for weakness in root.findall('.//cwe:Weakness', ns):
            weakness_id = weakness.get('ID')
            if not weakness_id:
                continue
                
            cwe_entity = f"CWE-{weakness_id}"
            
            related_weaknesses = weakness.find('.//cwe:Related_Weaknesses', ns)
            if related_weaknesses is not None:
                for related in related_weaknesses.findall('cwe:Related_Weakness', ns):
                    related_id = related.get('CWE_ID')
                    nature = related.get('Nature')
                    
                    if related_id and nature in ['ChildOf', 'ParentOf']:
                        related_entity = f"CWE-{related_id}"
                        
                        if nature == "ChildOf":
                            triples.append((cwe_entity, "is_child_of", related_entity))
                            triples.append((related_entity, "has_child", cwe_entity))
                        elif nature == "ParentOf":
                            triples.append((cwe_entity, "is_parent_of", related_entity))
                            triples.append((related_entity, "has_parent", cwe_entity))
    except Exception as e:
        print(f"Error parsing CWE: {e}")
    
    return triples

def parse_cwe_capec_relationships(xml_path: str):
    """Parse CWE-CAPEC attack pattern relationships."""
    triples = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Use the correct namespace from the XML
        ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}
        
        print(f"Using namespace: {ns}")
        
        # Look for Weakness elements
        weakness_elements = root.findall('.//cwe:Weakness', ns)
        print(f"Found {len(weakness_elements)} Weakness elements")
        
        for weakness in weakness_elements:
            weakness_id = weakness.get('ID')
            if not weakness_id:
                continue
                
            cwe_entity = f"CWE-{weakness_id}"
            
            # Look for Related_Attack_Patterns section (this is the correct name!)
            related_attack_patterns = weakness.find('.//cwe:Related_Attack_Patterns', ns)
            if related_attack_patterns is not None:
                pattern_elements = related_attack_patterns.findall('cwe:Related_Attack_Pattern', ns)
                print(f"Found {len(pattern_elements)} Related_Attack_Pattern elements for CWE-{weakness_id}")
                
                for attack_pattern in pattern_elements:
                    capec_id = attack_pattern.get('CAPEC_ID')
                    if capec_id:
                        capec_entity = f"CAPEC-{capec_id}"
                        triples.append((cwe_entity, "has_attack_pattern", capec_entity))
                        triples.append((capec_entity, "exploits_weakness", cwe_entity))
                        triples.append((cwe_entity, "related_to", capec_entity))
                        triples.append((capec_entity, "related_to", cwe_entity))
                        print(f"  Added: {cwe_entity} -> {capec_entity}")
                        
    except Exception as e:
        print(f"Error parsing CWE-CAPEC relationships: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Parsed {len(triples)} CWE-CAPEC relationships")
    return triples

def parse_capec_cwe_relationships(xml_path: str):
    """Parse CAPEC-CWE relationships from CAPEC XML."""
    triples = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Try different namespace approaches
        namespaces = [
            {'capec': 'http://capec.mitre.org/capec-3'},
            {'capec': 'http://capec.mitre.org/capec-2'},
            {}  # No namespace
        ]
        
        for ns in namespaces:
            print(f"Trying CAPEC namespace: {ns}")
            
            # Look for Attack_Pattern elements
            attack_patterns = root.findall('.//capec:Attack_Pattern', ns) if ns else root.findall('.//Attack_Pattern')
            print(f"Found {len(attack_patterns)} Attack_Pattern elements")
            
            for attack_pattern in attack_patterns:
                pattern_id = attack_pattern.get('ID')
                if not pattern_id:
                    continue
                    
                capec_entity = f"CAPEC-{pattern_id}"
                
                # Look for Related_Weaknesses section
                related_weaknesses = attack_pattern.find('.//capec:Related_Weaknesses', ns) if ns else attack_pattern.find('.//Related_Weaknesses')
                if related_weaknesses is not None:
                    weakness_elements = related_weaknesses.findall('capec:Related_Weakness', ns) if ns else related_weaknesses.findall('Related_Weakness')
                    print(f"Found {len(weakness_elements)} Related_Weakness elements for CAPEC-{pattern_id}")
                    
                    for weakness in weakness_elements:
                        cwe_id = weakness.get('CWE_ID')
                        if cwe_id:
                            cwe_entity = f"CWE-{cwe_id}"
                            triples.append((capec_entity, "exploits_weakness", cwe_entity))
                            triples.append((cwe_entity, "has_attack_pattern", capec_entity))
                            triples.append((capec_entity, "related_to", cwe_entity))
                            triples.append((cwe_entity, "related_to", capec_entity))
                            print(f"  Added: {capec_entity} -> {cwe_entity}")
            
            if triples:  # If we found relationships, break
                break
                        
    except Exception as e:
        print(f"Error parsing CAPEC-CWE relationships: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Parsed {len(triples)} CAPEC-CWE relationships")
    return triples

if __name__ == "__main__":
    create_simple_mitre_dataset()