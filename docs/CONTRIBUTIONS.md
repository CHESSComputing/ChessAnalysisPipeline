(contributions)=
# Contributions

We welcome feedback, suggestions, issues and bug reports.

Feel free to fork the [repository](https://github.com/CHESSComputing/ChessAnalysisPipeline) on GitHub and send a pull request, or to contribute by [submitting an issue](https://github.com/CHESSComputing/ChessAnalysisPipeline/issues). Each contribution helps CHAP grow and improve.

We appreciate your support!

## How to contribute

CHAP is an open source project and we encourage and gratefully receive feedback or contributions to improve our product. Please follow these guidelines to contribute to CHAP. Doing so grately helps us to address each contribution promptly and effectively.

There are many ways to contribute, from supplying examples, improving the documentation, submitting bug reports and feature requests or writing code that can be incorporated into CHAP itself.
Please [submit an issue](submit_chap_issue) to report a bug or request new features. Actual bug fixes or other types of contributions to the source code or documentation should be submitted after [creating a fork](create_chap_fork) and making a [pull request](make_chap_pull_request). Please do not use the issue tracker for support questions or to leave feedback. Instead, feel free to contact any of the administators directly.

Please only report a bug or request a new feature after checking the [existing open issues](https://github.com/CHESSComputing/ChessAnalysisPipeline/issues) and verifying that it has not yet been reported, requested or addressed!

(submit_chap_issue)=
### Submitting an issue

If you find a security vulnerability, do NOT open an issue. Contact any of the administators instead.

#### Reporting a bug

Report a bug by following these step to submit an issue:

1. Click on the green "New issues" botton on the [issues tab of the CHAP Github repository](https://github.com/CHESSComputing/ChessAnalysisPipeline/issues) to open a new issue.

1. Select the "CHAP CLI Bug report" option to open a template to provide us with information to help us track and reproduce the error.

1. After the "CHAP CLI Bug report" interface opens:

   a. Supply a clear and concise title to identify the issue.

   b. Edit and fill out the description form, being as specific as you can in answering each of the following items:

      - Give a clear and concise description of the nature of the bug.

      - Provide a minimal pipeline configuration in which the bug appears.

      - Describe what you expected to happen.

      - Describe what actually happens and if possible add a record or a screen shot of the (relevant part of the) CHAP logger output to show the error or explain the problem.

      - Provide information of the OS, CHAP version and python/conda environment.

      - Provide any additional information that can help in tracking down the issue.

   c. Click on the green "Create" botton on the right bottom corner of the dialogue interface.

#### Suggesting or requesting a new feature or enhancement

Request a new feature by following these step to submit an issue:

1. Click on the green "New issues" botton on the [issues tab of the CHAP Github repository](https://github.com/CHESSComputing/ChessAnalysisPipeline/issues) to open a new issue.

1. Select the "New PipelineItem request" option to open a template to provide us with information that helps us to modify existing or create new CHAP `PipelineItem`s to implement the feature.

1. After the "New PipelineItem request" dialogue interface opens:

   a. Supply a clear and concise title to describe the feature.

   b. Edit and fill out the description form, being as specific as you can in answering each of the following items:

      - Describe the intended use of the requested feature.

      - Describe the feature's inputs.

      - Describe the feature's optional interaction points.

      - Describe the feature's output.

      - Provide any additional information that can help in implementing the new feature.

   c. Click on the green "Create" botton on the right bottom corner of the dialogue interface.

#### Reporting any other issue

Follow these these step to submit any other issue, not covered by either option above:

1. Click on the green "New issues" botton on the [issues tab of the CHAP Github repository](https://github.com/CHESSComputing/ChessAnalysisPipeline/issues) to open a new issue.

1. Select the "Blank issue" option to open a template to provide us with information to help us resolve the issue.

1. After the new issue interface opens:

   a. Supply a clear and concise title to identify the issue,

   b. Edit and fill out the description form, being as specific as you can.

   c. Click on the green "Create" botton on the right bottom corner of the dialogue interface.

Check out the [Github documentation](https://docs.github.com/en) on [creating an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue) for additional helpful hints.

### Directly contributing to the source code or documentation

Actual bug fixes or other types of contributions to the source code or documentation should be submitted by making a pull request from your own fork.
Whether you are fixing bugs, adding features, or improving documentation, following the steps below ensures that any contributions can be verified and vetted by the administrators before incorporation into the main code base. To contribute directly to CHAP, you will need your own account on [Github](https://github.com).

(create_chap_fork)=
#### Create your own fork

Start by creating your own copy or "fork" of the CHAP repository before starting to contribute directly to the source code or documentation. This creates a copy of the repository that shares the code and documentation with the original CHAP repository, but that lets you make modifications or additions without directly affecting the source.

Create a fork by following these steps:

1. In a web browser, navigate to the [CHAP repository](https://github.com/CHESSComputing/ChessAnalysisPipeline) on GitHub.

1. In the top-right corner of the page, click "Fork".

1. After the "Create a new fork" interface opens:

   a. Select your own personal Github account under "Choose an owner". By default, forks are named the same as their upstream repositories. To follow this convention, leave the Repository name unchanged as "ChessAnalysisPipeline".

   b. Optionally, in the "Description" field, type a description of your fork.

   c. It is strongly recommended to select "Copy the main branch only" (its default setting).

   d. Click on the green "Create fork" button.

#### Clone your fork to your computer

After creating your own CHAP fork, you also need to create a local copy of the project or "clone" on your own computer. To clone your forked repositiory:

1. Click on the green "<> Code" botton.

1. Copy the URL for the repository: Under the "Local" tab choose to clone the repository using HTTPS or using an SSH key by selecting the respective tab and clicking the "Copy URL to clipboard" button to the right of the corresponding link. Choose HTTPS for a fast and easy setup if you just want to inspect the code base. Choose SSH if you plan to contribute frequently and want to avoid needing to enter your credentials for every push or pull.

1. On Mac or Linux, open a Terminal. On Windows, open Git Bash.

1. Change to the location where you want the cloned directory.

1. Type git clone, and then paste the URL you copied earlier. It should look like:
   ```bash
   $ git clone https://github.com/<your-Github-username>/ChessAnalysisPipeline.git
   ```
   or
   ```bash
   $ git clone git@github.com:<your-Github-username>/ChessAnalysisPipeline.git
   ```
   for cloning with HTTPS or SSH, respectively.

1. Press Enter

#### Create a branch to work on

It is STRONGLY recommended to create your own development branch instead of working directly with a fork of the main branch:

1. Navigate into the cloned CHAP repository directory:
   ```bash
   $ cd ChessAnalysisPipeline
   ```

2. Create your development branch:
   ```bash
   $ git branch BRANCH-NAME
   $ git checkout BRANCH-NAME
   ```
   after choosing a suitable name for BRANCH-NAME, e.g., "dev_<your_user_id>".

3. Create and track a copy of your newly created branch on your forked remote repository:
   ```bash
   $ git push --set-upstream origin BRANCH-NAME
   ```

#### Make and push changes

You can now add to or make changes to the code or documentation.

When you're ready to submit your changes, stage and commit your changes and push your changes to your forked remote repository on Github.
Best practice principles teach us to keep changes small, incremental and as logically grouped as possible within individual commits. There is no limit to the number of commits in any single pull request and it is much easier to review changes that are split across multiple commits.

Stage any files that are changed or added to be committed by executing:
```bash
$ git add <your-changed-files>
```
telling Git what to include in the next commit. When ready, commit these changes with:
```bash
$ git commit -m "a short description of the change"
```
and push your locally saved changes to your forked remote repository on Github:
```bash
$ git push
```

(make_chap_pull_request)=
#### Make a pull request

Making a pull request is the final step in contributing directly to the CHAP code base. After making changes or adding features to the code or documentation and pushing them to your forked remote Github repository you have to make a request to the CHAP administrators to have them included or "pulled" into the main code base.

To do return to your web browser and navigate again to the [CHAP repository](https://github.com/CHESSComputing/ChessAnalysisPipeline) on GitHub. Then:

1. You may see a banner indicating that your branch is ahead of the main CHAP branch or you may have to select your branch from the pull down box in the top left corner. Click the "contribute" button and the then the green "Open a pull request" button.

1. After Github brings you to the pull request interface: 

   a. Verify on the top that the base repository is CHESSComputing/ChessAnalysisPipeline and its base is "main", that the head repository is your branch on your forked repository and that Github display "Able to merge" in green right underneath this (contact the CHAP administrators if Github is not able to merge the changes).

   b. Add a clear and descriptive title for your pull request in the "Add a title" box.

   c. Add a clear description of your changes or additions "Add a description"
e" box. Making the title and description clear and consise will greatly help the administrators to determine whether your contribution is useful and needed for everyone to be included in the main code base.

   d. Click the green "Create pull request" button.

## Code review process

After submitting a pull request, changes will be reviewed by the administrators. Once approved, the pull request will be merged by the administrators and you can update your local branch by:

1. Navigating in your web browser to your forked CHAP repository.

1. Selecting your forked main branch from the pull down list on the top left, which should then say that it is ahead of CHESSComputing/ChessAnalysisPipeline:main.

1. Click the "Sync fork" button to synchronize or update your local main branch with that on the main CHAP repository.

1. Returning to your Terminal, navigating to the directory containing the clone of your forked repository and:

   a. Changing to your local main branch and pulling the updated remote one:
      ```bash
      $ git checkout main
      $ git pull
      ```

   b. Changing to your local development branch and merging the chamges from the main branch into your local development branch:
      ```bash
      $ git checkout BRANCH-NAME
      $ git merge main
      ```

At this point, you are fully synchronized with the main CHAP code base, ready for a new cycle of making and submitting changes.


Please, check the [Github documentation](https://docs.github.com/en) on [creating your own personal account](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github), [adding an SSH key to your account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account), [creating an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue), [forking a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo), or [submitting a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) for additional helpful hints.
