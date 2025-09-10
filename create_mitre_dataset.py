#!/usr/bin/env python3
"""
Unified MITRE dataset creation with support for direct/transitive relationship analysis.
"""

import xml.etree.ElementTree as ET
import os
import random
from collections import defaultdict

def create_mitre_dataset(include_hierarchical=True, mark_direct_relations=True):
    """
    Create MITRE dataset with proper direct/transitive relationship handling.
    
    Args:
        include_hierarchical: Whether to include hierarchical CWE/CAPEC relationships
        mark_direct_relations: Whether to mark direct CWE-CAPEC relationships with special tokens
    """
    print("Creating unified MITRE dataset...")
    
    # Parse CWE-CAPEC attack pattern relationships (DIRECT)
    print("Parsing direct CWE-CAPEC relationships...")
    cwe_capec_triples = parse_cwe_capec_relationships("mitre-data/raw/cwec_v4.17.xml")
    capec_cwe_triples = parse_capec_cwe_relationships("mitre-data/raw/capec_v3.9.xml")
    
    # Combine and deduplicate direct relationships
    direct_cwe_capec = list(set(cwe_capec_triples + capec_cwe_triples))
    
    # Get entities involved in direct relationships
    direct_entities = set()
    for h, r, t in direct_cwe_capec:
        direct_entities.add(h)
        direct_entities.add(t)
    
    print(f"Found {len(direct_cwe_capec)} direct CWE-CAPEC relationships")
    print(f"Entities in direct relationships: {len(direct_entities)}")
    
    all_triples = direct_cwe_capec.copy()
    
    # Add hierarchical relationships if requested
    if include_hierarchical:
        print("Adding hierarchical relationships...")
        capec_hier = parse_capec_hierarchical_filtered("mitre-data/raw/capec_v3.9.xml", direct_entities)
        cwe_hier = parse_cwe_hierarchical_filtered("mitre-data/raw/cwec_v4.17.xml", direct_entities)
        
        hierarchical_triples = capec_hier + cwe_hier
        all_triples.extend(hierarchical_triples)
        print(f"Added {len(hierarchical_triples)} hierarchical relationships")
    
    # Mark direct relationships with special relation types if requested
    if mark_direct_relations:
        marked_triples = []
        for h, r, t in all_triples:
            # Mark direct CWE-CAPEC relationships
            is_direct_cwe_capec = ((h.startswith('CWE-') and t.startswith('CAPEC-')) or
                                  (h.startswith('CAPEC-') and t.startswith('CWE-')))
            
            if is_direct_cwe_capec and r in ['has_attack_pattern', 'exploits_weakness', 'related_to']:
                # Add DIRECT_ prefix to mark these as direct relationships
                marked_relation = f"DIRECT_{r}"
                marked_triples.append((h, marked_relation, t))
            else:
                marked_triples.append((h, r, t))
        
        all_triples = marked_triples
        print("Marked direct CWE-CAPEC relationships with DIRECT_ prefix")
    
    # Create path-aware triples for multi-hop reasoning
    path_triples = create_path_aware_triples(all_triples)
    all_triples.extend(path_triples)
    
    # Show statistics
    print_dataset_statistics(all_triples)
    
    # Split data strategically
    train_triples, valid_triples, test_triples = split_dataset_strategically(all_triples)
    
    # Write dataset
    output_dir = "mitre_data"
    os.makedirs(output_dir, exist_ok=True)
    
    write_dataset_files(output_dir, train_triples, valid_triples, test_triples)
    
    print(f"\nUnified MITRE dataset created in {output_dir}/")
    print(f"Train: {len(train_triples)}, Valid: {len(valid_triples)}, Test: {len(test_triples)}")
    
    return train_triples, valid_triples, test_triples


def create_path_aware_triples(triples):
    """Create explicit 2-hop path triples to help model understand transitivity."""
    path_triples = []
    
    # Build adjacency for 2-hop paths
    adjacency = defaultdict(list)
    for h, r, t in triples:
        adjacency[h].append((r, t))
    
    # Find 2-hop paths and create explicit PATH relations
    for intermediate in adjacency:
        for r1, node1 in adjacency[intermediate]:
            for r2, node2 in adjacency.get(node1, []):
                # Create 2-hop path relation
                if intermediate != node2:  # Avoid self-loops
                    path_relation = f"PATH_{r1}_{r2}"
                    path_triples.append((intermediate, path_relation, node2))
    
    print(f"Created {len(path_triples)} explicit 2-hop path relationships")
    return path_triples


def split_dataset_strategically(all_triples):
    """Split dataset ensuring direct relationships are well represented in all splits."""
    # Separate direct and indirect relationships
    direct_triples = [t for t in all_triples if t[1].startswith('DIRECT_')]
    indirect_triples = [t for t in all_triples if not t[1].startswith('DIRECT_')]
    
    # Shuffle both groups
    random.shuffle(direct_triples)
    random.shuffle(indirect_triples)
    
    # Split direct relationships (70-15-15)
    direct_train_count = int(len(direct_triples) * 0.7)
    direct_valid_count = int(len(direct_triples) * 0.15)
    
    direct_train = direct_triples[:direct_train_count]
    direct_valid = direct_triples[direct_train_count:direct_train_count + direct_valid_count]
    direct_test = direct_triples[direct_train_count + direct_valid_count:]
    
    # Split indirect relationships (70-15-15)
    indirect_train_count = int(len(indirect_triples) * 0.7)
    indirect_valid_count = int(len(indirect_triples) * 0.15)
    
    indirect_train = indirect_triples[:indirect_train_count]
    indirect_valid = indirect_triples[indirect_train_count:indirect_train_count + indirect_valid_count]
    indirect_test = indirect_triples[indirect_train_count + indirect_valid_count:]
    
    # Combine and shuffle final splits
    train_triples = direct_train + indirect_train
    valid_triples = direct_valid + indirect_valid
    test_triples = direct_test + indirect_test
    
    random.shuffle(train_triples)
    random.shuffle(valid_triples)
    random.shuffle(test_triples)
    
    print(f"Split strategy: Direct({len(direct_train)}/{len(direct_valid)}/{len(direct_test)}), "
          f"Indirect({len(indirect_train)}/{len(indirect_valid)}/{len(indirect_test)})")
    
    return train_triples, valid_triples, test_triples


def print_dataset_statistics(all_triples):
    """Print comprehensive dataset statistics."""
    relation_counts = defaultdict(int)
    entity_types = defaultdict(int)
    
    for h, r, t in all_triples:
        relation_counts[r] += 1
        
        # Count entity types
        if h.startswith('CWE-'):
            entity_types['CWE'] += 1
        elif h.startswith('CAPEC-'):
            entity_types['CAPEC'] += 1
            
        if t.startswith('CWE-'):
            entity_types['CWE'] += 1
        elif t.startswith('CAPEC-'):
            entity_types['CAPEC'] += 1
    
    print(f"\nDataset Statistics:")
    print(f"Total triples: {len(all_triples)}")
    print(f"Unique entity mentions: CWE: {entity_types['CWE']}, CAPEC: {entity_types['CAPEC']}")
    
    print(f"\nRelation distribution:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel}: {count}")


def write_dataset_files(output_dir, train_triples, valid_triples, test_triples):
    """Write dataset files to disk."""
    for filename, triples in [("train.txt", train_triples), 
                             ("valid.txt", valid_triples), 
                             ("test.txt", test_triples)]:
        with open(os.path.join(output_dir, filename), 'w') as f:
            for h, r, t in triples:
                f.write(f"{h}\t{r}\t{t}\n")


def parse_cwe_capec_relationships(xml_path: str):
    """Parse CWE-CAPEC attack pattern relationships."""
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
            
            related_attack_patterns = weakness.find('.//cwe:Related_Attack_Patterns', ns)
            if related_attack_patterns is not None:
                for attack_pattern in related_attack_patterns.findall('cwe:Related_Attack_Pattern', ns):
                    capec_id = attack_pattern.get('CAPEC_ID')
                    if capec_id:
                        capec_entity = f"CAPEC-{capec_id}"
                        triples.append((cwe_entity, "has_attack_pattern", capec_entity))
                        triples.append((capec_entity, "exploits_weakness", cwe_entity))
                        triples.append((cwe_entity, "related_to", capec_entity))
                        triples.append((capec_entity, "related_to", cwe_entity))
                        
    except Exception as e:
        print(f"Error parsing CWE-CAPEC relationships: {e}")
    
    print(f"Parsed {len(triples)} CWE-CAPEC relationships from CWE XML")
    return triples


def parse_capec_cwe_relationships(xml_path: str):
    """Parse CAPEC-CWE relationships from CAPEC XML."""
    triples = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        namespaces = [
            {'capec': 'http://capec.mitre.org/capec-3'},
            {'capec': 'http://capec.mitre.org/capec-2'},
            {}
        ]
        
        for ns in namespaces:
            attack_patterns = root.findall('.//capec:Attack_Pattern', ns) if ns else root.findall('.//Attack_Pattern')
            
            if len(attack_patterns) > 0:
                for attack_pattern in attack_patterns:
                    pattern_id = attack_pattern.get('ID')
                    if not pattern_id:
                        continue
                        
                    capec_entity = f"CAPEC-{pattern_id}"
                    
                    related_weaknesses = attack_pattern.find('.//capec:Related_Weaknesses', ns) if ns else attack_pattern.find('.//Related_Weaknesses')
                    if related_weaknesses is not None:
                        for weakness in related_weaknesses.findall('capec:Related_Weakness', ns) if ns else related_weaknesses.findall('Related_Weakness'):
                            cwe_id = weakness.get('CWE_ID')
                            if cwe_id:
                                cwe_entity = f"CWE-{cwe_id}"
                                triples.append((capec_entity, "exploits_weakness", cwe_entity))
                                triples.append((cwe_entity, "has_attack_pattern", capec_entity))
                                triples.append((capec_entity, "related_to", cwe_entity))
                                triples.append((cwe_entity, "related_to", capec_entity))
                break
                        
    except Exception as e:
        print(f"Error parsing CAPEC-CWE relationships: {e}")
    
    print(f"Parsed {len(triples)} CAPEC-CWE relationships from CAPEC XML")
    return triples


def parse_capec_hierarchical_filtered(xml_path: str, allowed_entities: set):
    """Parse CAPEC hierarchical relationships for allowed entities only."""
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
            if capec_entity not in allowed_entities:
                continue
            
            related_patterns = attack_pattern.find('.//capec:Related_Attack_Patterns', ns)
            if related_patterns is not None:
                for related in related_patterns.findall('capec:Related_Attack_Pattern', ns):
                    related_id = related.get('CAPEC_ID')
                    nature = related.get('Nature')
                    
                    if related_id and nature in ['ChildOf', 'ParentOf']:
                        related_entity = f"CAPEC-{related_id}"
                        if related_entity in allowed_entities:
                            if nature == "ChildOf":
                                triples.append((capec_entity, "is_child_of", related_entity))
                                triples.append((related_entity, "has_child", capec_entity))
                            elif nature == "ParentOf":
                                triples.append((capec_entity, "is_parent_of", related_entity))
                                triples.append((related_entity, "has_parent", capec_entity))
    except Exception as e:
        print(f"Error parsing CAPEC hierarchical: {e}")
    
    return triples


def parse_cwe_hierarchical_filtered(xml_path: str, allowed_entities: set):
    """Parse CWE hierarchical relationships for allowed entities only."""
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
            if cwe_entity not in allowed_entities:
                continue
            
            related_weaknesses = weakness.find('.//cwe:Related_Weaknesses', ns)
            if related_weaknesses is not None:
                for related in related_weaknesses.findall('cwe:Related_Weakness', ns):
                    related_id = related.get('CWE_ID')
                    nature = related.get('Nature')
                    
                    if related_id and nature in ['ChildOf', 'ParentOf']:
                        related_entity = f"CWE-{related_id}"
                        if related_entity in allowed_entities:
                            if nature == "ChildOf":
                                triples.append((cwe_entity, "is_child_of", related_entity))
                                triples.append((related_entity, "has_child", cwe_entity))
                            elif nature == "ParentOf":
                                triples.append((cwe_entity, "is_parent_of", related_entity))
                                triples.append((related_entity, "has_parent", cwe_entity))
    except Exception as e:
        print(f"Error parsing CWE hierarchical: {e}")
    
    return triples


if __name__ == "__main__":
    random.seed(42)
    create_mitre_dataset(include_hierarchical=True, mark_direct_relations=True)
