require 'selenium-webdriver'
require 'nokogiri'
require 'fileutils'
require 'logger'
require 'json'
require 'thread'
require 'uri'
require 'timeout'

# Setup Selenium WebDriver
def setup_driver
  options = Selenium::WebDriver::Chrome::Options.new
  options.add_argument('--headless')
  options.add_argument('--disable-gpu')
  options.add_argument('--no-sandbox')
  options.add_argument('--disable-dev-shm-usage')
  options.add_argument('--disable-extensions')
  options.add_argument('--disable-infobars')
  options.add_argument('--window-size=1920,1080')
  options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36')

  driver = Selenium::WebDriver.for :chrome, options: options
  driver.manage.timeouts.page_load = 20 # Reduce timeout for faster response
  driver.manage.timeouts.script_timeout = 10
  return driver
end

# Search Yahoo
def search_yahoo(query, logger)
  driver = setup_driver()
  url = "https://search.yahoo.com/search?p=#{URI.encode_www_form_component(query)}"

  begin
    driver.get(url)
    logger.info("Yahoo search completed")

    doc = Nokogiri::HTML(driver.page_source)
    results = extract_yahoo_results(doc)
    driver.quit
    return results
  rescue => e
    logger.error("Yahoo failed: #{e.message}")
    driver.quit
    return nil
  end
end

# Extract Yahoo Search Results
def extract_yahoo_results(doc)
  results = []
  # Yahoo-specific selectors
  doc.css('div.dd.algo-sr h3.title a').each_with_index do |link, index|
    break if index >= 5 # Get 5 results for better chances
    title = link.text.strip
    href = link['href']
    
    # Skip sponsored, video, and shopping results
    next if href.nil? || title.empty?
    next if href.include?('youtube.com') || href.include?('shopping') || href.include?('sponsored')
    next if title.downcase.include?('sponsored') || title.downcase.include?('ad')

    results << { title: title, link: href }
  end
  
  # Fallback to general selectors if specific ones didn't work
  if results.empty?
    doc.css('h3 a').each_with_index do |link, index|
      break if index >= 5
      title = link.text.strip
      href = link['href']
      
      next if href.nil? || title.empty?
      next if href.include?('youtube.com') || href.include?('shopping') || href.include?('sponsored')
      
      results << { title: title, link: href }
    end
  end
  
  return results
end

# Search DuckDuckGo
def search_duckduckgo(query, logger)
  driver = setup_driver()
  url = "https://duckduckgo.com/?q=#{URI.encode_www_form_component(query)}"

  begin
    driver.get(url)
    # DuckDuckGo results load dynamically, so wait a bit
    sleep 2
    logger.info("DuckDuckGo search completed")

    doc = Nokogiri::HTML(driver.page_source)
    results = extract_duckduckgo_results(doc)
    driver.quit
    return results
  rescue => e
    logger.error("DuckDuckGo failed: #{e.message}")
    driver.quit
    return nil
  end
end

# Extract DuckDuckGo Search Results
def extract_duckduckgo_results(doc)
  results = []
  # DuckDuckGo-specific selectors
  doc.css('article h2 a').each_with_index do |link, index|
    break if index >= 5
    title = link.text.strip
    href = link['href']
    
    next if href.nil? || title.empty?
    next if href.include?('youtube.com') || href.include?('shopping') || href.include?('sponsored')
    
    results << { title: title, link: href }
  end
  
  # Fallback selectors
  if results.empty?
    doc.css('.result__title a').each_with_index do |link, index|
      break if index >= 5
      title = link.text.strip
      href = link['href']
      
      next if href.nil? || title.empty?
      next if href.include?('youtube.com') || href.include?('shopping')
      
      results << { title: title, link: href }
    end
  end
  
  # Second fallback for general links
  if results.empty?
    doc.css('a[data-testid*="result"]').each_with_index do |link, index|
      break if index >= 5
      title = link.text.strip
      href = link['href']
      
      next if href.nil? || title.empty?
      next if href.include?('youtube.com') || href.include?('shopping')
      
      results << { title: title, link: href }
    end
  end
  
  return results
end

# Run Both Searches in Parallel and Take the Fastest
def search_and_scrape(query)
  setup_directory()
  logger = setup_logger()
  
  # Store results from both engines
  search_results = { yahoo: nil, duckduckgo: nil }
  mutex = Mutex.new
  
  # Run searches in parallel
  threads = []
  threads << Thread.new do 
    yahoo_results = search_yahoo(query, logger)
    mutex.synchronize { search_results[:yahoo] = yahoo_results }
  end
  
  threads << Thread.new do
    ddg_results = search_duckduckgo(query, logger)
    mutex.synchronize { search_results[:duckduckgo] = ddg_results }
  end
  
  threads.each { |t| t.join(30) } # Wait up to 30 seconds
  
  # Select best results (non-nil, or with more results)
  best_results = if search_results[:yahoo].nil? && search_results[:duckduckgo].nil?
    logger.error("Both search engines failed.")
    []
  elsif search_results[:yahoo].nil?
    search_results[:duckduckgo]
  elsif search_results[:duckduckgo].nil?
    search_results[:yahoo]
  else
    # Choose the one with more results
    search_results[:yahoo].length >= search_results[:duckduckgo].length ? 
      search_results[:yahoo] : search_results[:duckduckgo]
  end
  
  # Scrape content from results
  results_data = []
  best_results.each_with_index do |result, index|
    break if index >= 3 # Limit to top 3 for speed
    
    begin
      scraped_content = scrape_content(result[:link], logger)
      next if scraped_content.empty?
      
      results_data << {
        title: result[:title],
        link: result[:link],
        content: scraped_content,
        scraped_at: Time.now.strftime("%Y-%m-%d %H:%M:%S")
      }
    rescue => e
      logger.error("Failed to process result #{result[:link]}: #{e.message}")
    end
  end
  
  store_to_json("data/search_results.json", results_data)
  store_to_txt("data/search_results.txt", results_data)
  
  logger.info("Scraping completed successfully with #{results_data.length} results.")
  return results_data
end

# Create a directory for storing results
def setup_directory
  FileUtils.mkdir_p('data')
end

# Initialize Logger
def setup_logger
  logger = Logger.new('data/scraper.log')
  logger.level = Logger::INFO
  return logger
end

# Function to scrape content from a given URL with better error handling
def scrape_content(url, logger)
  driver = setup_driver()
  content_lines = []
  visited_urls = Set.new # Track visited URLs to prevent loops
  max_pages = 2 # Limit to 2 pages maximum
  current_page = 0
  current_url = url
  
  begin
    Timeout.timeout(30) do # Overall timeout for the entire scraping process
      while current_page < max_pages && !visited_urls.include?(current_url)
        visited_urls.add(current_url)
        current_page += 1
        
        begin
          driver.get(current_url)
          sleep 1.5 # Brief wait for dynamic content
          
          doc = Nokogiri::HTML(driver.page_source)
          
          # Remove unwanted elements
          doc.css('script, style, footer, header, nav, noscript, iframe, 
                  .ads, .ad, .advertisement, .cookie-banner, .popup').remove
          
          # Try multiple content selectors
          main_content = nil
          ['article', 'main', '.content', '.post-content', '.article-content',
           '.entry-content', '#content', '.post', '.article'].each do |selector|
            main_content = doc.at_css(selector)
            break if main_content
          end
          
          # If none of the specific selectors worked, use the body
          main_content ||= doc.at_css('body')
          
          # Extract paragraphs and headings
          extracted_texts = main_content.css('h1, h2, h3, h4, p').map do |el|
            text = el.text.strip.gsub(/\s+/, ' ')
            text.length > 20 ? text : nil # Only keep substantial paragraphs
          end.compact
          
          content_lines += extracted_texts
          
          # Check for next page only if we haven't reached the limit
          if current_page < max_pages
            next_link = doc.at_css('a[rel="next"], a.next, .pagination a.next, .nav-next a')
            
            # Break if no next link found
            break unless next_link && next_link['href']
            
            # Construct absolute URL for next page
            next_url = begin
              URI.join(current_url, next_link['href']).to_s
            rescue
              nil
            end
            
            # Break if next URL is invalid or same as current
            break if next_url.nil? || next_url == current_url
            
            current_url = next_url
          else
            break # Max pages reached
          end
        rescue => e
          logger.error("Error on page #{current_page} of #{url}: #{e.message}")
          break # Exit the loop on error
        end
      end
    end
  rescue Timeout::Error
    logger.warn("Timeout reached while scraping #{url}")
  rescue => e
    logger.error("Failed to scrape #{url}: #{e.message}")
  ensure
    driver.quit
  end
  
  # Final filtering of content
  cleaned = content_lines.uniq.reject do |line|
    line.nil? || line.empty? || 
    line.length < 30 || # Filter out very short lines
    line.match?(/cookie|privacy policy|terms of service|all rights reserved|copyright/i) || 
    line.count('.') < 1 # At least one period for complete sentences
  end
  
  return cleaned
end

# Store data to JSON
def store_to_json(filename, data)
  File.open(filename, 'w') { |file| file.puts(JSON.pretty_generate(data)) }
end

# Store data to TXT
def store_to_txt(filename, data)
  File.open(filename, 'w') do |file|
    data.each do |result|
      file.puts "=== #{result[:title]} ==="
      file.puts "Source: #{result[:link]}"
      file.puts "Scraped: #{result[:scraped_at]}"
      file.puts "\n"
      file.puts result[:content].join("\n\n")
      file.puts "\n\n" + "="*50 + "\n\n" # Add separator between entries
    end
  end
end

# Get query from command line argument
query = ARGV[0]
if query.nil? || query.strip.empty?
  puts "Error: No search query provided."
  puts "Usage: ruby scraper.rb \"your search query\""
  exit 1
end

# Run scraper
begin
  results = search_and_scrape(query)
  puts "Scraping complete! Found #{results.length} results."
rescue => e
  puts "Error during scraping: #{e.message}"
  exit 1
end