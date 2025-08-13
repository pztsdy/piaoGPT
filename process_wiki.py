import json
from opencc import OpenCC

def process_wiki_json(json_path, output_file='lang_pool.txt', min_length=30):
    # 错误：cc = OpenCC({json_path})
    # 正确：使用 OpenCC 库自带的繁体到简体配置文件 't2s.json'
    try:
        cc = OpenCC('t2s') 
    except Exception as e:
        print(f"错误：OpenCC 初始化失败。请确保 'opencc-python-reimplemented' 及其配置文件已正确安装。")
        print(f"尝试运行 'pip uninstall opencc-python-reimplemented' 和 'pip install opencc-python-reimplemented'")
        print(f"详细错误信息: {e}")
        return # 如果 OpenCC 无法初始化，则退出函数
    
    count = 0
    print(f"开始处理 '{json_path}'，将输出到 '{output_file}'...")
    with open(json_path, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                article = json.loads(line)
                text = article.get('text', '')
                # 清理空行和标题
                lines = text.split('\n')
                cleaned_lines = [l.strip() for l in lines if l.strip()]
                simplified_text = cc.convert(" ".join(cleaned_lines)) # 使用空格连接
                if len(simplified_text) > min_length:
                    outfile.write(simplified_text + '\n')
                    count += 1
                    if count % 10000 == 0: # 调整打印频率
                        print(f"已处理 {count} 篇文章...")
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"处理行时发生错误: {e}, 行内容: {line[:100]}...")
                continue
    print(f"处理完成！共合并 {count} 篇文章到 {output_file}。")

if __name__ == '__main__':
    # 确保这里的路径是您实际的 JSON 文件路径
    process_wiki_json('G:\\wikipedia-lastest-chinese-zhcn-zhtw\\wikipedia-zh-cn-20250320.json')