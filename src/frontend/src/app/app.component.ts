import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpService } from './http.service';
import { CommonService } from './common.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
})
export class AppComponent implements OnInit {

  constructor(private httpService: HttpService, public commonService: CommonService) { }

  ngOnInit() {
    this.getSolved();
  }
  // TODO: Før oh med over alt

  private async getSolved() {
    this.commonService.state = await this.httpService.getSolved();
  }

  public async act(action: number) {
    this.commonService.state = await this.httpService.performAction(action);
  }
}
